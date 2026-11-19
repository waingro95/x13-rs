use std::{collections::BTreeSet, fs, io::Read, path::Path};

use anyhow::{Context, Result, bail, ensure};
use chrono::{Datelike, NaiveDate};
use serde::Deserialize;

use super::synthetic;
use crate::{Frequency, OracleConfiguration, Period, Series, series::MISSING_VALUE};

#[derive(Debug, Deserialize)]
pub(crate) struct SeriesManifest {
    pub(crate) series: Vec<SeriesMetadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SeriesMetadata {
    pub(crate) id: String,
    pub(crate) frequency: Frequency,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpecManifest {
    pub(crate) specs: Vec<SpecRow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpecRow {
    pub(crate) id: String,
    pub(crate) label: String,
    pub(crate) frequencies: Vec<Frequency>,
    #[serde(default)]
    pub(crate) requires_positive: bool,
    #[serde(default)]
    pub(crate) requires_nonnegative: bool,
    pub(crate) length: Lengths,
    /// Synthetic series this row runs on. Without it, the row runs on every
    /// applicable FRED series and never on a synthetic one.
    pub(crate) series: Option<Vec<String>>,
    pub(crate) configuration: OracleConfiguration,
    /// Calendar-specific arrays and explicit extension counts differ by frequency.
    pub(crate) quarterly_configuration: Option<OracleConfiguration>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Lengths {
    pub(crate) monthly: Option<LengthBounds>,
    pub(crate) quarterly: Option<LengthBounds>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LengthBounds {
    pub(crate) min: usize,
    pub(crate) max: usize,
}

impl SpecRow {
    pub(crate) fn select(&self, input: &Series) -> Result<Option<Series>> {
        let listed = match &self.series {
            Some(ids) => ids.contains(&input.series_id),
            None => !synthetic::is_synthetic(&input.series_id),
        };
        if !listed || !self.frequencies.contains(&input.frequency) {
            return Ok(None);
        }
        let bounds = match input.frequency {
            Frequency::Monthly => self.length.monthly.as_ref(),
            Frequency::Quarterly => self.length.quarterly.as_ref(),
            Frequency::Semiannual => None,
        }
        .context("missing applicable frequency length bounds")?;
        ensure!(
            bounds.min > 0 && bounds.min <= bounds.max,
            "invalid length bounds"
        );
        if input.values.len() < bounds.min {
            return Ok(None);
        }
        let mut offset = input.values.len().saturating_sub(bounds.max);
        // Build 62's year-span ceiling includes the leading partial calendar
        // year (editor.f Posffc/Ny). Long quarterly bounds reserve the row's
        // forecast/backcast span; short requested spans remain exactly sized.
        if input.frequency == Frequency::Quarterly && bounds.max >= 324 {
            while input.values.len() - offset
                + (usize::from(input.start.subperiod) - 1 + offset) % 4
                > bounds.max
            {
                offset += 1;
            }
        }
        let values = &input.values[offset..];
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite input");
        if (self.requires_positive && values.iter().any(|&v| v != MISSING_VALUE && v <= 0.))
            || (self.requires_nonnegative && values.iter().any(|&v| v != MISSING_VALUE && v < 0.))
        {
            return Ok(None);
        }
        let period = i64::from(input.frequency.periods_per_year());
        let date = i64::from(input.start.year) * period + i64::from(input.start.subperiod) - 1
            + i64::try_from(offset)?;
        Ok(Some(Series {
            series_id: input.series_id.clone(),
            frequency: input.frequency,
            start: Period::new(
                i32::try_from(date.div_euclid(period))?,
                u8::try_from(date.rem_euclid(period) + 1)?,
            ),
            values: values.to_vec(),
        }))
    }

    #[cfg(test)]
    pub(super) fn test_row(id: &str) -> Self {
        Self {
            id: id.into(),
            label: id.into(),
            frequencies: vec![Frequency::Monthly],
            requires_positive: false,
            requires_nonnegative: false,
            length: Lengths {
                monthly: Some(LengthBounds { min: 1, max: 780 }),
                quarterly: None,
            },
            series: None,
            configuration: OracleConfiguration::direct_x11(),
            quarterly_configuration: None,
        }
    }
}

pub(crate) fn load(root: &Path) -> Result<(Vec<SpecRow>, Vec<Series>)> {
    let specs: SpecManifest = read_toml(&root.join("specs.toml"))?;
    let metadata: SeriesManifest = read_toml(&root.join("series.toml"))?;
    ensure!(!metadata.series.is_empty(), "empty series manifest");
    let mut ids = BTreeSet::new();
    let mut series = metadata
        .series
        .iter()
        .map(|meta| {
            ensure!(
                valid_id(&meta.id) && !synthetic::is_synthetic(&meta.id) && ids.insert(&meta.id),
                "invalid or duplicate series id {}",
                meta.id
            );
            let path = root.join("series").join(format!("{}.csv", meta.id));
            let file = fs::File::open(&path).with_context(|| path.display().to_string())?;
            read_csv(&meta.id, meta.frequency, file).with_context(|| path.display().to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let path = root.join("synthetic.toml");
    let manifest: synthetic::Manifest = read_toml(&path)?;
    let mut generated = BTreeSet::new();
    for descriptor in &manifest.series {
        let id = descriptor.id();
        ensure!(
            valid_id(id) && synthetic::is_synthetic(id) && generated.insert(id.to_owned()),
            "invalid or duplicate synthetic series id {id}"
        );
        let input = descriptor
            .generate(&series)
            .with_context(|| format!("{}: {id}", path.display()))?;
        series.push(input);
    }
    validate_specs(&specs.specs, &generated)?;
    Ok((specs.specs, series))
}

pub(crate) fn read_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    toml::from_str(&fs::read_to_string(path).with_context(|| path.display().to_string())?)
        .with_context(|| path.display().to_string())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}

fn validate_specs(specs: &[SpecRow], synthetic: &BTreeSet<String>) -> Result<()> {
    ensure!(!specs.is_empty(), "empty spec manifest");
    let mut ids = BTreeSet::new();
    for spec in specs {
        ensure!(
            valid_id(&spec.id) && ids.insert(&spec.id),
            "invalid or duplicate spec id {}",
            spec.id
        );
        if let Some(series) = &spec.series {
            ensure!(
                !series.is_empty() && series.iter().all(|id| synthetic.contains(id)),
                "row {} must list known synthetic series only",
                spec.id
            );
        }
        ensure!(
            !spec.label.is_empty() && !spec.frequencies.is_empty(),
            "empty spec label or frequencies"
        );
        for (i, frequency) in spec.frequencies.iter().enumerate() {
            ensure!(
                !spec.frequencies[..i].contains(frequency),
                "duplicate frequency"
            );
            let bounds = match frequency {
                Frequency::Monthly => spec.length.monthly.as_ref(),
                Frequency::Quarterly => spec.length.quarterly.as_ref(),
                Frequency::Semiannual => bail!("parity supports monthly and quarterly only"),
            }
            .context("missing length bounds")?;
            ensure!(
                bounds.min > 0 && bounds.min <= bounds.max,
                "invalid length bounds for {}",
                spec.id
            );
        }
    }
    Ok(())
}

fn read_csv(id: &str, frequency: Frequency, input: impl Read) -> Result<Series> {
    ensure!(
        frequency != Frequency::Semiannual,
        "unsupported corpus frequency"
    );
    let mut reader = csv::Reader::from_reader(input);
    let headers = reader.headers()?;
    ensure!(
        headers.len() == 2
            && matches!(&headers[0], "observation_date" | "DATE")
            && &headers[1] == id,
        "expected FRED date/{id} columns"
    );
    let step = 12 / u32::from(frequency.periods_per_year());
    let mut start = None;
    let mut previous = None;
    let mut values = Vec::new();
    for record in reader.records() {
        let record = record?;
        let date = NaiveDate::parse_from_str(&record[0], "%Y-%m-%d")?;
        ensure!(
            date.day() == 1 && date.month0().is_multiple_of(step),
            "date not on a period boundary"
        );
        let month = i64::from(date.year()) * 12 + i64::from(date.month0());
        ensure!(
            previous.is_none_or(|last| month > last),
            "duplicate or unordered FRED dates"
        );
        if let Some(last) = previous {
            let skipped = usize::try_from((month - last) / i64::from(step) - 1)?;
            ensure!(skipped <= 12000, "implausibly large FRED date gap");
            values.extend(std::iter::repeat_n(MISSING_VALUE, skipped));
        }
        previous = Some(month);
        start.get_or_insert(Period::new(date.year(), (date.month0() / step + 1) as u8));
        let value: f64 = match record[1].trim() {
            "" | "." => MISSING_VALUE,
            text => text.parse()?,
        };
        ensure!(value.is_finite(), "nonfinite FRED observation");
        values.push(value);
    }
    Ok(Series {
        series_id: id.into(),
        frequency,
        start: start.context("empty FRED series")?,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Frequency, Period};

    #[test]
    fn every_row_renders_a_spec_for_each_of_its_frequencies() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity");
        let (rows, series) = load(&root).unwrap();
        assert!(series.iter().any(|s| s.values.contains(&MISSING_VALUE)));
        for row in &rows {
            for frequency in &row.frequencies {
                let input = series
                    .iter()
                    .filter(|s| s.frequency == *frequency)
                    .find_map(|s| row.select(s).unwrap())
                    .unwrap_or_else(|| panic!("{}: no {frequency:?} series", row.id));
                let config = if *frequency == Frequency::Quarterly {
                    row.quarterly_configuration
                        .as_ref()
                        .unwrap_or(&row.configuration)
                } else {
                    &row.configuration
                };
                let rendered = crate::spec_file::render(&input, config)
                    .unwrap_or_else(|e| panic!("{}/{}: {e}", row.id, input.series_id));
                assert!(rendered.contains("file = \"input.dat\""));
                let seats = config.method() == crate::Method::Seats;
                assert_eq!(rendered.contains("seats {"), seats, "{}", row.id);
                assert_eq!(rendered.contains("x11 {"), !seats, "{}", row.id);
            }
        }
    }

    #[test]
    fn fred_csv_keeps_exact_values_and_rejects_invalid_data() {
        let series = read_csv(
            "RATE",
            Frequency::Quarterly,
            b"observation_date,RATE\n2000-01-01,-0.0\n2000-04-01,1.0000000000000002\n".as_slice(),
        )
        .unwrap();
        assert_eq!(series.start, Period::new(2000, 1));
        assert_eq!(series.values[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(series.values[1].to_bits(), 1.0000000000000002_f64.to_bits());
        for bad in [
            "observation_date,RATE\n2000-02-01,1\n",
            "observation_date,RATE\n2000-01-01,NaN\n",
            "observation_date,WRONG\n2000-01-01,1\n",
            "observation_date,RATE\n",
        ] {
            assert!(
                read_csv("RATE", Frequency::Quarterly, bad.as_bytes()).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn gaps_become_missing_codes_and_pass_a_positive_only_row() {
        let series = read_csv(
            "RATE",
            Frequency::Quarterly,
            b"observation_date,RATE\n2000-01-01,1\n2000-04-01,.\n2000-07-01,\n2001-01-01,2\n"
                .as_slice(),
        )
        .unwrap();
        assert_eq!(series.values, [1., -99999., -99999., -99999., 2.]);
        let rust = series.rust_values();
        assert_eq!((rust[0], rust[4]), (1., 2.));
        assert!(rust[1..4].iter().all(|v| v.is_nan()));
        assert_eq!(crate::series::build_62_values(&rust), series.values);
        let mut row = SpecRow::test_row("positive");
        row.frequencies = vec![Frequency::Quarterly];
        row.length.quarterly = Some(LengthBounds { min: 1, max: 100 });
        row.requires_positive = true;
        assert_eq!(row.select(&series).unwrap().unwrap().values, series.values);
        assert!(
            read_csv(
                "RATE",
                Frequency::Quarterly,
                b"observation_date,RATE\n2000-04-01,1\n2000-01-01,2\n".as_slice()
            )
            .is_err()
        );
    }

    #[test]
    fn a_row_takes_the_longest_applicable_tail_of_a_series() {
        let spec: SpecRow = toml::from_str(
            r#"
            id = "x11-multiplicative"
            label = "x11: multiplicative"
            frequencies = ["monthly", "quarterly"]
            requires_positive = true
            [length.monthly]
            min = 3
            max = 3
            [length.quarterly]
            min = 4
            max = 4
            [configuration]
            kind = "direct_x11"
            mode = "multiplicative"
            seasonal_filter = "s3x5"
        "#,
        )
        .unwrap();
        let mut input = Series {
            series_id: "S".into(),
            frequency: Frequency::Monthly,
            start: Period::new(1999, 11),
            values: vec![-1., 0., 1., 2., 3.],
        };
        let case = spec.select(&input).unwrap().unwrap();
        assert_eq!(case.start, Period::new(2000, 1));
        assert_eq!(case.values, [1., 2., 3.]);
        input.values[4] = 0.;
        assert!(spec.select(&input).unwrap().is_none());
        input.values = vec![1., 2.];
        assert!(spec.select(&input).unwrap().is_none());
    }

    #[test]
    fn rows_select_synthetic_series_only_by_listing_them() {
        let input = |id: &str| Series {
            series_id: id.into(),
            frequency: Frequency::Monthly,
            start: Period::new(2000, 1),
            values: vec![1.; 3],
        };
        let mut row = SpecRow::test_row("row");
        assert!(row.select(&input("FRED")).unwrap().is_some());
        assert!(row.select(&input("SYN_A")).unwrap().is_none());
        row.series = Some(vec!["SYN_A".into()]);
        assert!(row.select(&input("SYN_A")).unwrap().is_some());
        assert!(row.select(&input("SYN_B")).unwrap().is_none());
        assert!(row.select(&input("FRED")).unwrap().is_none());
    }

    #[test]
    fn rows_need_unique_ids_valid_bounds_and_known_synthetic_series() {
        let none = BTreeSet::new();
        let mut rows = vec![SpecRow::test_row("one"), SpecRow::test_row("one")];
        assert!(validate_specs(&rows, &none).is_err());
        rows.pop();
        rows[0].length.monthly = Some(LengthBounds { min: 10, max: 2 });
        assert!(validate_specs(&rows, &none).is_err());
        rows[0].length.monthly = Some(LengthBounds { min: 1, max: 2 });
        assert!(validate_specs(&rows, &none).is_ok());
        let known = BTreeSet::from(["SYN_A".to_owned()]);
        for (listed, valid) in [
            (vec![], false),
            (vec!["SYN_B"], false),
            (vec!["SYN_A"], true),
        ] {
            rows[0].series = Some(listed.into_iter().map(str::to_owned).collect());
            assert_eq!(validate_specs(&rows, &known).is_ok(), valid);
        }
    }
}
