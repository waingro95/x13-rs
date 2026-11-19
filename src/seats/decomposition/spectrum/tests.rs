use super::*;

#[test]
fn denominator_floor_keeps_the_sign() {
    let floor = f64::from_bits(0x3d3c_25c2_6849_7682);
    for (input, expected) in [
        (0.0, floor),
        (-0.0, -floor),
        (floor.next_down(), floor),
        ((-floor).next_up(), -floor),
        (floor, floor),
        (-floor, -floor),
        (floor.next_up(), floor.next_up()),
    ] {
        assert_eq!(clamp_denominator(input).to_bits(), expected.to_bits());
    }
    for denominator in [0.0, -0.0, floor, -floor] {
        let denominator = [denominator];
        let spectrum = Spectrum::new(SpectrumComponent::Seasonal, &[1.0], &denominator);
        // The sum starts from a positive zero, which absorbs a negative one.
        let expected = if denominator[0] == 0.0 {
            1.0 / floor
        } else {
            1.0 / denominator[0]
        };
        assert_eq!(spectrum.evaluate(0.0).to_bits(), expected.to_bits());
    }
}

#[test]
fn only_a_differenced_trend_floors_the_search_angle() {
    let trend = Spectrum::new(SpectrumComponent::Trend, &[1.0], &[1.0]);
    let stationary =
        Spectrum::new(SpectrumComponent::Trend, &[1.0], &[1.0]).with_differencing(false);
    let seasonal = Spectrum::new(SpectrumComponent::Seasonal, &[1.0], &[1.0]);
    let cycle = Spectrum::new(SpectrumComponent::Cycle, &[1.0], &[1.0]);
    let two_pi = f64::from_bits(0x4019_21fb_5444_2d11);
    assert_eq!(
        TREND_ANGLE_FLOOR.to_bits(),
        ((two_pi * 2.0) / 360.0).to_bits()
    );
    assert_eq!(GRID_STEP.to_bits(), (BUILD_62_PI / 100000.0).to_bits());
    for angle in [-TREND_ANGLE_FLOOR, 0.0, TREND_ANGLE_FLOOR.next_down()] {
        assert_eq!(
            trend.evaluation_angle(angle).to_bits(),
            TREND_ANGLE_FLOOR.to_bits()
        );
        for raw in [&stationary, &seasonal, &cycle] {
            assert_eq!(raw.evaluation_angle(angle).to_bits(), angle.to_bits());
        }
    }
    for angle in [TREND_ANGLE_FLOOR, TREND_ANGLE_FLOOR.next_up()] {
        assert_eq!(trend.evaluation_angle(angle).to_bits(), angle.to_bits());
    }
}

#[test]
fn a_constant_spectrum_is_its_own_minimum() {
    for component in [
        SpectrumComponent::Seasonal,
        SpectrumComponent::Trend,
        SpectrumComponent::Cycle,
    ] {
        let spectrum = Spectrum::new(component, &[1.0], &[1.0]);
        let local = local_minimum(&spectrum, 0.0, -1000.0, 1000.0);
        assert_eq!(local.to_bits(), 1.0_f64.to_bits());
        for frequency in [Frequency::Quarterly, Frequency::Monthly] {
            let grid = grid_minimum(&spectrum, frequency);
            assert_eq!(grid.to_bits(), 1.0_f64.to_bits());
        }
    }
}

#[test]
fn searches_find_the_minimum_of_a_cosine() {
    // 2 + cos(x) falls to 1 at pi; 2 - cos(x) falls to 1 at zero.
    let falling = Spectrum::new(SpectrumComponent::Cycle, &[2.0, 1.0], &[1.0]);
    let upper = global_minimum(&falling, 0.5 * BUILD_62_PI, BUILD_62_PI);
    assert!((upper - 1.0).abs() < 1e-6, "{upper}");
    let grid = grid_minimum(&falling, Frequency::Monthly);
    assert!((grid - 1.0).abs() < 1e-8, "{grid}");

    let rising = Spectrum::new(SpectrumComponent::Cycle, &[2.0, -1.0], &[1.0]);
    let lower = global_minimum(&rising, 0.0, 0.5 * BUILD_62_PI);
    assert!((lower - 1.0).abs() < 1e-6, "{lower}");
    assert_eq!(
        grid_minimum(&rising, Frequency::Monthly).to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn the_trend_grid_starts_two_degrees_above_zero() {
    let rising = Spectrum::new(SpectrumComponent::Trend, &[2.0, -1.0], &[1.0]);
    assert_eq!(
        grid_minimum(&rising, Frequency::Monthly).to_bits(),
        rising.evaluate(TREND_ANGLE_FLOOR).to_bits()
    );
}
