//! The log-additive mode of X-11. The series is decomposed additively in
//! logs, then the seasonal factors and the trend are exponentiated and the
//! trend is corrected for the bias of the transformation: Build 62 `logar.f`,
//! `antilg.f` and `trbias.f`, called from `x11pt1.f` to `x11pt3.f`.

use crate::error::{
    NumericalDetail, X11LogAdditiveStage as Stage, X11LogAdditiveUnavailableReason as Reason,
};
use crate::time::AxisSpan;
use crate::{Error, numerics, x11};

/// What a log-additive run carries between its stages: the logged series and
/// the irregular the bias correction reads.
pub(super) struct Phase {
    dates: AxisSpan,
    logged: Vec<f64>,
    /// The pass-C irregular (table C13) over the whole extended span.
    /// `trbias.f` sums its squares over the backcasts and forecasts too.
    c13: Vec<f64>,
}

impl Phase {
    /// `logar.f`: takes the log of every value. A value that is not positive,
    /// or a log that is not finite, is an error dated at that observation.
    pub(super) fn new(values: &[f64], dates: AxisSpan) -> Result<Self, Error> {
        let mut phase = Self {
            dates,
            logged: Vec::with_capacity(values.len()),
            c13: Vec::new(),
        };
        for (index, &value) in values.iter().enumerate() {
            if value <= 0.0 {
                return Err(phase.error(Stage::InputLog, index, index, Reason::NonPositiveInput));
            }
            let value = value.ln();
            phase.finite(value, Stage::InputLog, index)?;
            phase.logged.push(value);
        }
        Ok(phase)
    }

    /// The logged series.
    pub(super) fn logged(&self) -> &[f64] {
        &self.logged
    }

    /// Keeps the pass-C irregular for [`Self::finish_trend`].
    pub(super) fn retain_c13(&mut self, c13: &[f64]) {
        self.c13 = c13.to_vec();
    }

    /// `antilg.f`: exponentiates the seasonal factors in place. A factor that
    /// overflows, or underflows to zero, is an error.
    pub(super) fn exponentiate_seasonal(&self, seasonal: &mut [f64]) -> Result<(), Error> {
        for (index, value) in seasonal.iter_mut().enumerate() {
            self.finite(*value, Stage::SeasonalExponentiation, index)?;
            *value = value.exp();
            self.finite(*value, Stage::SeasonalExponentiation, index)?;
            if *value == 0.0 {
                return Err(self.error(
                    Stage::SeasonalExponentiation,
                    index,
                    index,
                    Reason::ZeroSeasonalFactor,
                ));
            }
        }
        Ok(())
    }

    /// Exponentiates the log trend in place and applies the `trbias.f` bias
    /// correction: every value is multiplied by `exp(sum(c13^2) / 2n)` and by
    /// the Henderson-smoothed seasonal factor. A trend the correction leaves
    /// zero or negative is returned as it is, for the caller to replace.
    pub(super) fn finish_trend(
        &self,
        trend: &mut [f64],
        seasonal: &[f64],
        period: usize,
        endpoints: x11::SevenTermEndpoints,
    ) -> Result<(), Error> {
        debug_assert_eq!(self.c13.len(), trend.len());
        for (index, value) in trend.iter_mut().enumerate() {
            self.finite(*value, Stage::TrendExponentiation, index)?;
            *value = value.exp();
            self.finite(*value, Stage::TrendExponentiation, index)?;
        }
        let sum = self.irregular_square_sum()?;
        let common = (sum / (2.0 * self.c13.len() as f64)).exp();
        if !common.is_finite() {
            return Err(self.error(
                Stage::BiasFactor,
                0,
                trend.len() - 1,
                Reason::NonFiniteValue,
            ));
        }
        // trbias.f smooths the seasonal factors with a Henderson filter of
        // two years less one term at an I/C ratio of 4.5, with the requested
        // seven-term endpoints. The ratio the filter returns is not the
        // trend's, so it is dropped.
        let (mut bias, _) = numerics::henderson_filter(seasonal, 2 * period - 1, 4.5, endpoints);
        self.apply_bias(common, &mut bias, trend)
    }

    fn irregular_square_sum(&self) -> Result<f64, Error> {
        let mut sum = 0.0;
        for &value in &self.c13 {
            // gfortran fuses the square into the running sum in trbias.f.
            sum = value.mul_add(value, sum);
            if !sum.is_finite() {
                return Err(self.error(
                    Stage::IrregularVariance,
                    0,
                    self.c13.len() - 1,
                    Reason::NonFiniteValue,
                ));
            }
        }
        Ok(sum)
    }

    fn apply_bias(&self, common: f64, bias: &mut [f64], trend: &mut [f64]) -> Result<(), Error> {
        for (index, (bias, trend)) in bias.iter_mut().zip(trend).enumerate() {
            // trbias.f forms the bias factor first and then multiplies it
            // into the trend; `(trend * common) * bias` rounds differently.
            *bias *= common;
            self.finite(*bias, Stage::BiasFactor, index)?;
            *trend *= *bias;
            self.finite(*trend, Stage::TrendBias, index)?;
        }
        Ok(())
    }

    fn finite(&self, value: f64, stage: Stage, index: usize) -> Result<(), Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(self.error(stage, index, index, Reason::NonFiniteValue))
        }
    }

    fn error(&self, stage: Stage, first: usize, last: usize, reason: Reason) -> Error {
        Error::from(NumericalDetail::X11LogAdditive {
            stage,
            first: self.dates.period(first),
            last: self.dates.period(last),
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeAxis;

    /// Three years of ones with a zero irregular.
    fn phase(period: usize) -> Phase {
        let axis = if period == 12 {
            TimeAxis::monthly(1999, 11)
        } else {
            TimeAxis::quarterly(1999, 3)
        }
        .unwrap();
        let mut phase = Phase::new(&vec![1.0; period * 3], axis.span(period * 3).unwrap()).unwrap();
        phase.retain_c13(&vec![0.0; period * 3]);
        phase
    }

    fn assert_error(
        error: Error,
        phase: &Phase,
        stage: Stage,
        first: usize,
        last: usize,
        reason: Reason,
    ) {
        assert_eq!(
            error,
            Error::from(NumericalDetail::X11LogAdditive {
                stage,
                first: phase.dates.period(first),
                last: phase.dates.period(last),
                reason,
            })
        );
    }

    #[test]
    fn input_is_logged_and_bad_values_are_dated() {
        let dates = TimeAxis::quarterly(1999, 3).unwrap().span(4).unwrap();
        let values = [1.0, 2.0, 4.0, 8.0];
        let phase = Phase::new(&values, dates).unwrap();
        assert_eq!(
            phase.logged,
            [0.0, std::f64::consts::LN_2, 4_f64.ln(), 8_f64.ln()]
        );
        for index in [0, 3] {
            let mut values = values;
            values[index] = 0.0;
            assert_error(
                Phase::new(&values, dates).err().unwrap(),
                &phase,
                Stage::InputLog,
                index,
                index,
                Reason::NonPositiveInput,
            );
            values[index] = f64::INFINITY;
            assert_error(
                Phase::new(&values, dates).err().unwrap(),
                &phase,
                Stage::InputLog,
                index,
                index,
                Reason::NonFiniteValue,
            );
        }
    }

    #[test]
    fn seasonal_overflow_and_underflow_are_errors() {
        let phase = phase(4);
        for index in [0, 11] {
            for (value, reason) in [
                (1000.0, Reason::NonFiniteValue),
                (-1000.0, Reason::ZeroSeasonalFactor),
            ] {
                let mut seasonal = vec![0.0; 12];
                seasonal[index] = value;
                assert_error(
                    phase.exponentiate_seasonal(&mut seasonal).unwrap_err(),
                    &phase,
                    Stage::SeasonalExponentiation,
                    index,
                    index,
                    reason,
                );
            }
        }
    }

    #[test]
    fn nonfinite_log_trend_is_an_error() {
        // Negative infinity would exponentiate to a finite zero.
        let phase = phase(4);
        for index in [0, 11] {
            let mut trend = [0.0; 12];
            trend[index] = f64::NEG_INFINITY;
            assert_error(
                phase
                    .finish_trend(
                        &mut trend,
                        &[1.0; 12],
                        4,
                        x11::SevenTermEndpoints::LegacyFiveTerm,
                    )
                    .unwrap_err(),
                &phase,
                Stage::TrendExponentiation,
                index,
                index,
                Reason::NonFiniteValue,
            );
        }
    }

    #[test]
    fn bias_uses_the_whole_retained_irregular() {
        for period in [4, 12] {
            let mut phase = phase(period);
            let count = phase.logged.len();
            let mut c13 = vec![0.0; count];
            c13[0] = 1.0;
            c13[count - 1] = 2.0;
            phase.retain_c13(&c13);
            // A zero log trend exponentiates to 1, so the finished trend is
            // the bias factor itself.
            let mut bias = vec![0.0; count];
            phase
                .finish_trend(
                    &mut bias,
                    &vec![1.0; count],
                    period,
                    x11::SevenTermEndpoints::LegacyFiveTerm,
                )
                .unwrap();
            let expected = (5.0 / (2.0 * count as f64)).exp();
            for bias in bias {
                assert!((bias - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn overflow_is_reported_by_stage() {
        let mut phase = phase(4);
        for index in [0, 11] {
            let mut trend = vec![0.0; 12];
            trend[index] = 1000.0;
            assert_error(
                phase
                    .finish_trend(
                        &mut trend,
                        &[1.0; 12],
                        4,
                        x11::SevenTermEndpoints::LegacyFiveTerm,
                    )
                    .unwrap_err(),
                &phase,
                Stage::TrendExponentiation,
                index,
                index,
                Reason::NonFiniteValue,
            );
        }
        for (value, stage) in [
            (f64::MAX, Stage::IrregularVariance),
            (100.0, Stage::BiasFactor),
        ] {
            phase.retain_c13(&[value; 12]);
            assert_error(
                phase
                    .finish_trend(
                        &mut [0.0; 12],
                        &[1.0; 12],
                        4,
                        x11::SevenTermEndpoints::LegacyFiveTerm,
                    )
                    .unwrap_err(),
                &phase,
                stage,
                0,
                11,
                Reason::NonFiniteValue,
            );
        }
    }

    #[test]
    fn zero_and_negative_trends_are_returned() {
        let phase = phase(4);
        let mut zero = [-1000.0; 12];
        phase
            .finish_trend(
                &mut zero,
                &[1.0; 12],
                4,
                x11::SevenTermEndpoints::LegacyFiveTerm,
            )
            .unwrap();
        assert!(zero.iter().all(|&value| value == 0.0));
        // Positive seasonal factors can have negative Henderson extrapolations.
        let seasonal = [1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut bias = [0.0; 12];
        phase
            .finish_trend(
                &mut bias,
                &seasonal,
                4,
                x11::SevenTermEndpoints::HendersonSevenTerm,
            )
            .unwrap();
        assert!(bias.iter().any(|&value| value < 0.0));
    }

    #[test]
    fn bias_filter_follows_the_requested_endpoints() {
        let phase = phase(4);
        let seasonal = [
            0.7, 1.2, 0.9, 1.3, 0.75, 1.15, 0.95, 1.25, 0.8, 1.1, 1.0, 1.2,
        ];
        let mut legacy = [0.0; 12];
        phase
            .finish_trend(
                &mut legacy,
                &seasonal,
                4,
                x11::SevenTermEndpoints::LegacyFiveTerm,
            )
            .unwrap();
        let mut henderson = [0.0; 12];
        phase
            .finish_trend(
                &mut henderson,
                &seasonal,
                4,
                x11::SevenTermEndpoints::HendersonSevenTerm,
            )
            .unwrap();
        assert_ne!(legacy[0].to_bits(), henderson[0].to_bits());
        assert_eq!(legacy[3..9], henderson[3..9]);
        // The symmetric five-term weights at this position do not depend on
        // the I/C ratio.
        let expected = (-0.07342657342657342 * seasonal[0])
            + 0.2937062937062937 * seasonal[1]
            + 0.5594405594405595 * seasonal[2]
            + 0.2937062937062937 * seasonal[3]
            - 0.07342657342657342 * seasonal[4];
        assert!((legacy[2] - expected).abs() < 1e-15);
    }

    #[test]
    fn square_sum_is_fused_and_bias_products_are_not() {
        let mut phase = phase(4);
        phase.retain_c13(&[0.7, 1.3, 2.1, 0.7, 1.3, 2.1, 0.7, 1.3, 2.1, 0.7, 1.3, 2.1]);
        // The expected bits are exact decimal arithmetic on these inputs,
        // rounded once per term.
        assert_eq!(
            phase.irregular_square_sum().unwrap().to_bits(),
            0x403a_5c28_f5c2_8f5d
        );
        let separate = phase
            .c13
            .iter()
            .fold(0.0, |sum, &value| sum + value * value);
        assert_eq!(separate.to_bits(), 0x403a_5c28_f5c2_8f5c);
        let mut bias = [1.3];
        let mut trend = [1.4918246976412703];
        phase.apply_bias(2.99916325, &mut bias, &mut trend).unwrap();
        assert_eq!(trend[0].to_bits(), 0x4017_4416_e2af_0832);
        assert_ne!(
            trend[0].to_bits(),
            ((1.4918246976412703_f64 * 2.99916325) * 1.3).to_bits()
        );
    }

    #[test]
    fn bias_overflow_is_dated_at_its_observation() {
        let phase = phase(4);
        for index in [0, 11] {
            let mut bias = [1.0; 12];
            bias[index] = f64::MAX;
            assert_error(
                phase
                    .apply_bias(2.0, &mut bias, &mut [1.0; 12])
                    .unwrap_err(),
                &phase,
                Stage::BiasFactor,
                index,
                index,
                Reason::NonFiniteValue,
            );
            let mut trend = [1.0; 12];
            trend[index] = f64::MAX;
            assert_error(
                phase
                    .apply_bias(2.0, &mut [1.0; 12], &mut trend)
                    .unwrap_err(),
                &phase,
                Stage::TrendBias,
                index,
                index,
                Reason::NonFiniteValue,
            );
        }
    }

    #[test]
    fn bias_filter_spans_two_years_less_one() {
        for period in [4, 12] {
            let dates = TimeAxis::monthly(2000, 1)
                .unwrap()
                .span(period * 6)
                .unwrap();
            let mut phase = Phase::new(&vec![1.0; period * 6], dates).unwrap();
            phase.retain_c13(&vec![0.0; period * 6]);
            let mut seasonal = vec![1.0; period * 6];
            let center = period * 3;
            seasonal[center] = 2.0;
            let mut bias = vec![0.0; period * 6];
            phase
                .finish_trend(
                    &mut bias,
                    &seasonal,
                    period,
                    x11::SevenTermEndpoints::HendersonSevenTerm,
                )
                .unwrap();
            // The filter reaches `period - 1` terms from its center and no
            // further.
            assert!((bias[center + period - 1] - 1.0).abs() > 1e-5);
            assert!((bias[center + period] - 1.0).abs() < 1e-15);
        }
    }
}
