# CLAUDE.md

@CONTRIBUTING.md

Notes for coding agents, on top of `CONTRIBUTING.md`:

- Work in the Nix dev shell. Check `IN_NIX_SHELL` once and reuse the shell
  rather than wrapping each command in `nix develop -c`.
- Run the parity rows a change affects with `PARITY_FILTER` before the full
  table, which is slow.
- Keep Build 62's operation order in numerical code. A mathematically
  equivalent rewrite usually fails the table.
- Never change tolerances, call-table rows or deviations to make a case pass.
- Write standard rustdoc. Leave design history and rejected alternatives to
  git, not to comments.
