use super::BoundCandidate::{AlreadyFixed, Bound, Skip};
use super::{BoundFixing, BoundRule, SearchResult, search};
use crate::Frequency;
use crate::seats::Orders;
use crate::seats::ar1_seasonal_ma::residual::Ar1SeasonalMaObjective;
use crate::seats::refit::test_support::{alternating, differenced_noise, noise, random_walk};
use crate::seats::residual::AirlineObjective;
use crate::seats::{Error, Failure, Stage};

/// SEARCH for the quarterly (0,1,1)(0,1,1) on the differenced series `wd`.
fn airline(wd: &[f64]) -> Result<SearchResult<2>, Error> {
    search(
        AirlineObjective::new(Frequency::Quarterly, wd),
        [0.1, 0.2],
        [false; 2],
        BoundRule::from_orders(Orders::new([0, 1, 1], [0, 1, 1])),
    )
    .map(|(result, _)| result)
}

/// SEARCH for the quarterly (1,1,0)(0,1,1) on the differenced series `wd`.
fn ar1_seasonal_ma(wd: &[f64]) -> Result<SearchResult<2>, Error> {
    search(
        Ar1SeasonalMaObjective::new(Frequency::Quarterly, wd, true),
        [0.1, 0.2],
        [false; 2],
        BoundRule::from_orders(Orders::new([1, 1, 0], [0, 1, 1])),
    )
    .map(|(result, _)| result)
}

#[test]
fn two_free_coordinates_converge_with_a_covariance() {
    let result = airline(&noise(64)).unwrap();
    assert_eq!(result.parameters, [0.1119499645912974, 0.15025635224233705]);
    assert_eq!(result.bound_mask, [false, false]);
    assert_eq!(result.fixed, 0);
    assert!(!result.all_fixed);
    assert_eq!(result.objective, 4.527261477260789);
    assert_eq!(result.determinant_scale, 1.0008128468072255);
    assert_eq!(result.evaluations, 19);
    assert_eq!(result.scaled_residuals.len(), 69);
    assert_eq!(result.scaled_residuals[0], -0.0003655959454438653);
    assert_eq!(result.scaled_residuals[68], 0.26567090037570845);
    let covariance = result.covariance.unwrap();
    assert_eq!(
        covariance.standard_errors,
        [0.12380709680270267, 0.12429231668060825]
    );
    assert_eq!(
        covariance.correlation,
        [[1.0, -0.15876373955490217], [-0.15876373955490217, 1.0]]
    );
}

#[test]
fn ar_and_seasonal_ma_coordinates_converge_together() {
    let result = ar1_seasonal_ma(&noise(64)).unwrap();
    assert_eq!(
        result.parameters,
        [-0.11640132062364679, 0.14799555814694193]
    );
    assert_eq!(result.bound_mask, [false, false]);
    assert_eq!(result.objective, 4.525626973735117);
    assert_eq!(result.determinant_scale, 1.0007032983108672);
    assert_eq!(result.evaluations, 13);
    assert_eq!(result.scaled_residuals.len(), 67);
    assert_eq!(result.scaled_residuals[0], 7.69513054064338e-5);
    assert_eq!(result.scaled_residuals[66], 0.26329462217523847);
    let covariance = result.covariance.unwrap();
    assert_eq!(
        covariance.standard_errors,
        [0.12585006424303047, 0.12388720684968389]
    );
    assert_eq!(
        covariance.correlation,
        [[1.0, 0.1551638804482677], [0.1551638804482677, 1.0]]
    );
}

#[test]
fn a_coordinate_near_its_upper_bound_is_fixed_there() {
    // The seasonal MA coordinate ends on its bound; SEARCH goes on with the
    // regular one alone and reports 100 for the fixed one.
    let result = airline(&differenced_noise(64, 4)).unwrap();
    assert_eq!(result.parameters, [0.12983905347510014, 0.99]);
    assert_eq!(result.bound_mask, [false, true]);
    assert_eq!(result.fixed, 1);
    assert!(!result.all_fixed);
    assert_eq!(result.objective, 5.7308250523479);
    assert_eq!(result.determinant_scale, 1.0874121168303523);
    assert_eq!(result.evaluations, 22);
    let covariance = result.covariance.unwrap();
    assert_eq!(covariance.standard_errors, [0.1206673201933292, 100.0]);
    assert_eq!(covariance.correlation, [[1.0, 100.0], [100.0, 100.0]]);
}

#[test]
fn a_regular_ar_coordinate_near_its_lower_bound_is_fixed_at_minus_one() {
    let result = ar1_seasonal_ma(&alternating(64)).unwrap();
    assert_eq!(result.parameters, [-1.0, 0.21717928250567514]);
    assert_eq!(result.bound_mask, [true, false]);
    assert_eq!(result.fixed, 1);
    assert_eq!(result.objective, 0.40284637330172035);
    assert_eq!(result.determinant_scale, 1.001535001025622);
    assert_eq!(result.evaluations, 17);
    let covariance = result.covariance.unwrap();
    assert_eq!(covariance.standard_errors, [100.0, 0.1180863828994147]);
    assert_eq!(covariance.correlation, [[100.0, 100.0], [100.0, 1.0]]);
}

#[test]
fn search_ends_without_a_covariance_when_every_coordinate_is_fixed() {
    // Noise differenced at lags 1 and 4 puts both MA coordinates on the
    // upper bound.
    let e = noise(69);
    let wd = (5..69)
        .map(|t| e[t] - e[t - 1] - e[t - 4] + e[t - 5])
        .collect::<Vec<_>>();
    let result = airline(&wd).unwrap();
    assert_eq!(result.parameters, [0.99, 0.99]);
    assert_eq!(result.bound_mask, [true, true]);
    assert!(result.all_fixed);
    assert_eq!(result.fixed, 1);
    assert_eq!(result.objective, 6.663627296651141);
    assert_eq!(result.determinant_scale, 1.1473629074839953);
    assert_eq!(result.evaluations, 13);
    assert!(result.covariance.is_none());
}

#[test]
fn search_stops_at_the_iteration_limit() {
    let result = airline(&random_walk(64)).unwrap();
    assert_eq!(
        result.parameters,
        [-0.8553392783623397, -0.44861455047806176]
    );
    assert_eq!(result.objective, 11.896669310338986);
    assert_eq!(result.determinant_scale, 1.0140329854008021);
    assert_eq!(result.evaluations, 83);
    let covariance = result.covariance.unwrap();
    assert_eq!(
        covariance.standard_errors,
        [0.05959106432051192, 0.12098119984185148]
    );
}

#[test]
fn a_zero_series_is_an_error_where_build_62_reads_nv_unwritten() {
    assert_eq!(
        airline(&[0.0; 64]).err(),
        Some(Failure::UnwrittenSearchCount { objective: 0.0 }.at(Stage::Refit))
    );
}

fn ar2() -> BoundFixing {
    BoundFixing::new(BoundRule::from_orders(Orders::new([2, 1, 0], [0, 1, 1])))
}

#[test]
fn with_d_one_only_the_first_regular_ar_coordinate_is_fixed_at_minus_one() {
    let mut f = ar2();
    assert_eq!(f.candidate(0, false, -0.99), Bound(-1.0));
    assert_eq!(f.candidate(1, false, -0.99), Skip);
    assert_eq!(f.regular_lower_count, 1);
    assert_eq!(f.candidate(2, false, -0.99), Bound(-0.99));
    let mut f = ar2();
    assert_eq!(f.candidate(0, false, 0.3), Skip);
    assert_eq!(f.candidate(1, false, -0.99), Bound(-1.0));
}

#[test]
fn lower_fix_depends_on_d_and_bd() {
    // (0,1,0)(1,1,0): seasonal AR at its lower bound stays free with BD=1.
    let mut f = BoundFixing::new(BoundRule::from_orders(Orders::new([0, 1, 0], [1, 1, 0])));
    assert_eq!(f.candidate(0, false, -0.99), Skip);
    // With BD=0 it is fixed at -UR and does not count as a regular unit root.
    let mut f = BoundFixing::new(BoundRule::from_orders(Orders::new([1, 1, 0], [1, 0, 0])));
    assert_eq!(f.candidate(0, false, -0.99), Bound(-1.0));
    assert_eq!(f.candidate(1, false, -0.99), Bound(-1.0));
    assert_eq!(f.regular_lower_count, 1);
    // D=0 allows two regular unit roots and D=2 none; MA keeps XMIN.
    let mut f = BoundFixing::new(BoundRule::from_orders(Orders::new([2, 0, 0], [0, 0, 0])));
    assert_eq!(f.candidate(0, false, -0.99), Bound(-1.0));
    assert_eq!(f.candidate(1, false, -0.99), Bound(-1.0));
    let mut f = BoundFixing::new(BoundRule::from_orders(Orders::new([1, 2, 1], [0, 0, 0])));
    assert_eq!(f.candidate(0, false, -0.99), Skip);
    assert_eq!(f.candidate(1, false, -0.99), Bound(-0.99));
}

#[test]
fn lower_fix_depends_on_the_coordinate_kind() {
    // (3,d,1)(1,bd,1): three regular AR, one seasonal AR, then the
    // regular and seasonal MA.
    let rule = |d, bd| BoundRule::from_orders(Orders::new([3, d, 1], [1, bd, 1]));
    // D=1, BD=1: one regular unit root, BPHI stays free, MA to XMIN.
    let mut f = BoundFixing::new(rule(1, 1));
    let fixes = (0..6)
        .map(|j| f.candidate(j, false, -0.99))
        .collect::<Vec<_>>();
    assert_eq!(
        fixes,
        [Bound(-1.0), Skip, Skip, Skip, Bound(-0.99), Bound(-0.99)]
    );
    let mut f = BoundFixing::new(rule(0, 0));
    let fixes = (0..4)
        .map(|j| f.candidate(j, false, -0.99))
        .collect::<Vec<_>>();
    assert_eq!(fixes, [Bound(-1.0), Bound(-1.0), Skip, Bound(-1.0)]);
    let mut f = BoundFixing::new(rule(2, 1));
    assert_eq!(f.candidate(0, false, -0.99), Skip);
    assert_eq!(f.candidate(3, false, 0.99), Bound(0.99));
}

#[test]
fn a_coordinate_held_from_the_start_does_not_count_as_a_unit_root() {
    for held in [0, 1] {
        let mut f = ar2();
        assert_eq!(f.candidate(held, true, -1.0), AlreadyFixed);
        assert_eq!(f.regular_lower_count, 0);
        assert_eq!(f.candidate(1 - held, false, -0.99), Bound(-1.0));
    }
}

#[test]
fn an_upper_fix_does_not_count_as_a_unit_root() {
    let mut f = ar2();
    assert_eq!(f.candidate(0, false, 0.99), Bound(0.99));
    assert_eq!(f.candidate(1, false, -0.99), Bound(-1.0));
}

#[test]
fn a_coordinate_left_free_on_its_bound_stays_in_the_free_count() {
    let mut f = ar2();
    let mut free_count = 3;
    let first = f.candidate(0, false, -0.99);
    assert_eq!(first, Bound(-1.0));
    first.decrement_free(&mut free_count);
    assert_eq!(free_count, 2);
    let second = f.candidate(1, false, -0.99);
    assert_eq!(second, Skip);
    second.decrement_free(&mut free_count);
    assert_eq!(free_count, 2);
    assert_eq!(f.regular_lower_count, 1);
    f.candidate(2, true, 0.99).decrement_free(&mut free_count);
    assert_eq!(free_count, 1);
}
