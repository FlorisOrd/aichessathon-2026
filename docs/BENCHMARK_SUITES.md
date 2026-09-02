# Fixed benchmark suites

The Phase 2 benchmark corpus is generated from the public Lichess evaluation database, a CC0
collection of positions evaluated by Stockfish. The committed repository contains only the small
FEN suites and their provenance metadata; it does not contain the source database.

## Reproducible selection

`tools.build_position_suite` scans an explicit one-based record window of the documented JSONL
stream; its defaults remain the original records 1 through 250,000. Use `--start-record` and
`--records` to select another window without changing how records are numbered.
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

## Sealed promotion suites

Three engine-independent 96-position suites are pre-generated for later promotion decisions:

- `promotion-a-96.fens`: source records 250,001 through 500,000; seed 20,260,902.
- `promotion-b-96.fens`: source records 500,001 through 750,000; seed 20,260,903.
- `promotion-c-96.fens`: source records 750,001 through 1,000,000; seed 20,260,904.

Each suite contains 24 opening/high-material, 48 middlegame, and 24 endgame positions, with 48
positions for each side to move and 48 positions with an immediately available capture. Pawn
structures are unique within each suite. The source windows are disjoint from one another and from
the original development/holdout source window. `promotion-suite-metadata.json` records each exact
window's scanned-byte count and SHA-256, the suite SHA-256, selection counts, structural statistics,
and the per-position evaluation provenance.

Promotion A is the next available promotion validation set. Once it is used for a promotion
decision, it is consumed and Promotion B becomes next. Promotion C is reserved after B. A consumed
promotion suite must never be reused to tune a candidate. Generating and structurally validating
these suites uses python-chess only; no project agent is run against them.

Regenerate all three in one forward-only source pass with:

```powershell
uv run --with zstandard python -m tools.build_promotion_suites
```
