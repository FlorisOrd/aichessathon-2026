# Fixed benchmark suites

The Phase 2 benchmark corpus is generated from the public Lichess evaluation database, a CC0
collection of positions evaluated by Stockfish. The committed repository contains only the small
FEN suites and their provenance metadata; it does not contain the source database.

## Reproducible selection

`tools.build_position_suite` scans a fixed 250,000-record prefix of the documented JSONL stream.
It takes the first principal variation from the highest-depth evaluation for each record and keeps
neutral, valid, non-terminal standard-chess positions. Selection is independent of this project's
engine. A fixed-seed reservoir sample balances game phase, side to move, capture availability, and
castling-rights presence. Every source position is normalized to a six-field FEN with halfmove `0`
and fullmove `1`; the source has no move stack or repetition history.

The metadata records the source URL and dump date, all criteria, examined and eligible counts,
the SHA-256 of the exact scanned JSONL bytes, the generated file hashes, and each selected
position's Stockfish evaluation, depth, phase, and source record number.

To regenerate directly from Lichess without adding a project dependency:

```powershell
uv run --with zstandard python -m tools.build_position_suite
```

If the large source file is cached, keep it outside the repository (especially outside OneDrive)
and pass its path with `--source`.

## Development and holdout policy

Use `development-fast-16.fens` for quick iteration and `development-64.fens` for routine evidence.
Both are development data; repeatedly inspecting their results is expected.

`holdout-32.fens` is an independently assigned deterministic one-third split within every sampling
stratum. Do not inspect its per-position or aggregate game results and do not tune against it during
routine development. Use it only for infrequent promotion decisions after a candidate has already
passed the development suite. The holdout was not selected based on results from this engine.

Changing the seed, source window, eligibility rules, strata, or split creates a new suite version;
do not silently replace these files while comparing experiments.
