# ARIMA root solver

`src/arima/roots/solver.rs` finds the roots of the ARIMA factor polynomials,
which have degree 1 to 3. Build 62 uses RPOLY (ACM TOMS Algorithm 493) for
this. The Rust solver is not a translation of it. This note records where the
solver comes from and what it must do.

## Provenance

The solver is an independent implementation, built from the closed-form
quadratic and cubic formulas and Newton's method. Its choice of deflation
direction follows W. Kahan, "To Solve a Real Cubic Equation" (1986), which the
code cites where the choice is made. It is not derived from RPOLY, from
Build 62's `rpoly.f` or from the source of any other root finder. RPOLY carries
the ACM software licence, so work on the solver from this note and the cited
paper, and do not consult RPOLY or ports of it.

`src/seats/roots.rs` is a different case: it translates the root finder in
Build 62's SEATS code.

## Interface

```rust
/// Roots of c[0]·z^n + c[1]·z^(n-1) + … + c[n], for n in 1..=3.
/// The caller guarantees c[0] != 0, finite coefficients and n <= 3.
pub(super) fn polynomial_roots(descending: &[f64]) -> Result<Vec<ComplexRoot>, RootComputationError>;
```

`roots()` in `src/arima/roots/polynomial.rs` builds `descending` from ARIMA
coefficients, strips leading zeros and calls the solver. Two functions use the
result: `has_strictly_inside_root` (is any root's modulus strictly below 1?)
and `detailed_roots` in `details.rs` (the modulus and frequency of each root,
for the unit-root checks of automatic modelling).

## Output contract

1. Exactly n roots, all finite. Return `Err(RootComputationError)` when an
   intermediate value is nonfinite. Never panic or loop forever.
2. Never return `Ok` with roots that underflow has corrupted. If an
   intermediate of a quadratic or cubic underflows to zero or a subnormal and
   that changes the result, rescale exactly or return `Err`. Two rescalings
   are exact: multiplying every coefficient by 2^t, and substituting
   z = 2^s·w, solving for w and scaling the roots back by 2^s. Harmless
   underflow needs no action, for example b·b underflowing while 4·a·c is
   normal. A polynomial whose intermediates don't underflow is never rescaled.
   Overflow returns `Err`.
3. Degree 1: the single division `-c[1] / c[0]`.
4. Degree 2, with a = c[0], b = c[1], c = c[2]:
   - Compute the discriminant D = b·b − 4·a·c directly in `f64`. If D is
     nonfinite, return `Err`; the polynomial `[-1.0, -1e308, -1.0]` is an
     example.
   - D ≥ 0: use the cancellation-free form q = −(b + sign(b)·√D)/2 and the
     roots q/a and c/q, with sign(0) = +1. When q = 0 (b = 0 and D = 0), both
     roots are 0.
   - D < 0: the pair −b/(2a) ± i·√(−D)/(2|a|).
5. Degree 3: take one real root from the closed form, the trigonometric form
   when there are three real roots and Cardano's form otherwise. Polish it
   with Newton steps on the cubic, stopping when |p(x)| stops decreasing or
   after 8 steps. Divide the cubic by (z − root) synthetically and solve the
   quadratic by rule 4. Polish each real root of the quadratic the same way on
   the undeflated cubic.
6. A real root has `imaginary.to_bits() == 0` (+0.0). `details.rs` takes
   `atan2(imaginary, real)`, so −0.0 would flip the sign of a frequency.
7. A complex pair is adjacent, with the +imaginary root first, identical real
   parts, and imaginary parts that are exact negations.
8. Order: ascending modulus, then ascending real part, then descending
   imaginary part, compared with `f64::total_cmp`. A finite root whose squared
   modulus overflows sorts after every other root and is not an error. For
   example `has_strictly_inside_root(&[1e150, 1e-10])` is `Ok(true)`, because
   the root near 1e-150 comes first.
9. Deterministic: identical inputs give bit-identical outputs, with no mutable
   process-wide state.

## Checks

The unit tests in `src/arima/roots/` cover the contract. Two kinds of input
are worth knowing when changing the method:

- Scale (rule 2). Each of these returns either `Err` or roots within 1e-14
  relative error of the true roots: `[1, 0, 1e-110, 1e-170]`,
  `[1, 1e-200, 0, 0]`, `[1e-200, -3e-200, 2e-200]`, `[1e-200, 0, 1e-200]`,
  `[1e-200, 0, -1e-200]`, `[1e-310, 1e-310, 1e-310]`,
  `[1e-200, -6e-200, 11e-200, -6e-200]`, `[1, -6e-100, 11e-200, -6e-300]` and
  `[-1.25e211, -2.35e127, 7.3e84, -1]`.
- Accuracy. Both of these reconstruct each coefficient to within 1e-14
  relative error: `[1, -3.000001, 2.2500025000000003, -0.500001]`, which is
  (z−0.5)(z−0.500001)(z−2), and `[1, -999999.5, -499999.3, -700000]`, which is
  (z−1e6)(z²+0.5z+0.7). Plain Horner evaluation and forward-only deflation
  each fail one of them.

Because the method differs from RPOLY, a decision that depends on a root lying
exactly on the unit circle can differ from Build 62's.
`tests/parity/deviations.toml` lists the known cases.
