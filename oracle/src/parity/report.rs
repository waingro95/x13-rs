use std::fmt::Write;

use super::{
    case::{CaseResult, Status},
    rejection_map::{self, RejectionMap},
};
use crate::Frequency;

// Both-reject cases count as passes, and their own columns show how many
// passes compared no output. "Off map" counts the both-reject cases,
// at either frequency, whose Rust kind the rejection map does not allow for
// Build 62's message, or whose message it lacks. With the map
// enforced they have already failed, so they count among the failures and
// leave the both-reject columns; otherwise they still pass.
pub(crate) fn render(
    rows: &[(&str, &str, Vec<Frequency>)],
    cases: &[CaseResult],
    map: &RejectionMap,
) -> String {
    let off_map = |case: &CaseResult| match &case.status {
        Status::BothRejected { fortran, kind, .. } => map.check(fortran, *kind).misses(),
        Status::Fail(failures) => failures
            .iter()
            .any(|failure| failure.output == rejection_map::FAILURE_OUTPUT),
        _ => false,
    };
    let mut report = String::from(
        "| Call | Monthly passing | Quarterly passing | Monthly both-reject | Quarterly both-reject | Both-reject off map |\n|---|---:|---:|---:|---:|---:|\n",
    );
    for &(id, label, ref frequencies) in rows {
        let relevant = |frequency| {
            cases.iter().filter(move |c| {
                c.spec_id == id
                    && c.frequency == frequency
                    && !matches!(c.status, Status::Nondeterministic)
            })
        };
        let cell = |frequency| {
            if !frequencies.contains(&frequency) {
                return "—".into();
            }
            let passes = relevant(frequency).filter(|c| c.status.passes()).count();
            format!("{passes} / {}", relevant(frequency).count())
        };
        let both = |frequency| {
            if !frequencies.contains(&frequency) {
                return "—".into();
            }
            let count = relevant(frequency)
                .filter(|c| matches!(c.status, Status::BothRejected { .. }))
                .count();
            count.to_string()
        };
        let off = [Frequency::Monthly, Frequency::Quarterly]
            .into_iter()
            .map(|frequency| relevant(frequency).filter(|c| off_map(c)).count())
            .sum::<usize>();
        writeln!(
            report,
            "| {} | {} | {} | {} | {} | {off} |",
            label.replace('|', "\\|"),
            cell(Frequency::Monthly),
            cell(Frequency::Quarterly),
            both(Frequency::Monthly),
            both(Frequency::Quarterly)
        )
        .unwrap();
    }
    let counted = cases
        .iter()
        .filter(|c| !matches!(c.status, Status::Nondeterministic));
    let both = counted
        .clone()
        .filter(|c| matches!(c.status, Status::BothRejected { .. }))
        .count();
    let off = counted.clone().filter(|c| off_map(c)).count();
    writeln!(
        report,
        "\nTotal: {} / {} passing, {both} of them both-reject; {off} both-reject case(s) off the \
         rejection map.",
        counted.clone().filter(|c| c.status.passes()).count(),
        counted.count()
    )
    .unwrap();
    for (id, _, _) in rows {
        let deviations = cases
            .iter()
            .filter(|c| c.spec_id == *id && matches!(c.status, Status::Deviation(_)))
            .count();
        if deviations > 0 {
            writeln!(report, "\n{id}: {deviations} deviation pass(es).").unwrap();
        }
    }
    let nondeterministic = cases
        .iter()
        .filter(|c| matches!(c.status, Status::Nondeterministic))
        .count();
    if nondeterministic > 0 {
        writeln!(
            report,
            "\n{nondeterministic} fortran-nondeterministic case(s) excluded from denominators."
        )
        .unwrap();
    }
    report.push_str("\nFirst 50 failures:\n");
    let mut count = 0;
    for case in cases {
        if let Status::Fail(failures) = &case.status
            && let Some(failure) = failures.first()
        {
            if count == 50 {
                return report;
            }
            writeln!(
                report,
                "- `{}` {} index {:?}: Fortran={}, Rust={}, relative error={:?}",
                case.id,
                failure.output,
                failure.index,
                failure.expected,
                failure.actual,
                failure.relative_error
            )
            .unwrap();
            count += 1;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nondeterministic_cases_are_excluded_and_deviations_are_counted() {
        let both = |id: &str, frequency, fortran: &str, kind| {
            CaseResult::test(
                id,
                frequency,
                Status::BothRejected {
                    fortran: fortran.into(),
                    rust: "TypedRejection".into(),
                    kind,
                },
            )
        };
        let cases = vec![
            CaseResult::test("A/row", Frequency::Monthly, Status::Pass),
            CaseResult::test("B/row", Frequency::Monthly, Status::Deviation(vec![0])),
            CaseResult::test("C/row", Frequency::Monthly, Status::Nondeterministic),
            CaseResult::test("D/row", Frequency::Quarterly, Status::Fail(vec![])),
            both(
                "E/row",
                Frequency::Monthly,
                "ERROR: example",
                x13::ErrorKind::InvalidInput,
            ),
            // Off the map: outside the allowed kinds, and unmapped.
            both(
                "F/row",
                Frequency::Monthly,
                "ERROR: example",
                x13::ErrorKind::Numerical,
            ),
            both(
                "G/row",
                Frequency::Quarterly,
                "ERROR: other",
                x13::ErrorKind::Numerical,
            ),
        ];
        let map = RejectionMap::parse(
            "[[rejections]]\nprefix = \"ERROR: example\"\nallowed = [\"InvalidInput\"]\n\
             routine = \"r\"\nnote = \"n\"\n",
        )
        .unwrap();
        let rows = [(
            "row",
            "Example",
            vec![Frequency::Monthly, Frequency::Quarterly],
        )];
        let report = render(&rows, &cases, &map);
        assert!(report.contains("| Example | 4 / 4 | 1 / 2 | 2 | 1 | 2 |"));
        assert!(report.contains(
            "Total: 5 / 6 passing, 3 of them both-reject; 2 both-reject case(s) off the \
             rejection map."
        ));
        assert!(report.contains("1 deviation"));
        assert!(report.contains("1 fortran-nondeterministic"));

        // Enforced, F and G fail on the map and still count as off it.
        let mut enforced = cases;
        for case in &mut enforced {
            rejection_map::apply(case, &map, true);
        }
        let report = render(&rows, &enforced, &map);
        assert!(
            report.contains("| Example | 3 / 4 | 0 / 2 | 1 | 0 | 2 |"),
            "{report}"
        );
        assert!(report.contains(
            "Total: 3 / 6 passing, 1 of them both-reject; 2 both-reject case(s) off the \
             rejection map."
        ));
        assert!(report.contains("`F/row` rejection map"));
    }
}
