#[test]
fn parity() -> anyhow::Result<()> {
    x13_oracle::parity::run_from_environment()
}
