# Numba search v1 speed port

Semantic oracle: exact killer/history `6686f7ca8412776247149512a28fe7ec18f034f9`.
Base release: `3227c4a93750033a227b48882f057a77f7c937ae`.
Core imported by cherry-picking reviewed prototype `6a89bccbf58bc1cb7aa7b007e8238849b820e93d`.

`engine.py` remains byte-for-byte the Python oracle. `numba_engine.py` ports its tapered integer
evaluation, promotion/MVV-LVA ordering, UCI tie order, two killers, capped depth-squared history,
negamax alpha-beta, captures/promotions qsearch, all check evasions, and mate-distance scores.
No TT, pruning heuristic, evaluator change, pondering, or new chess feature is included.
`harness/` is unchanged from base release `3227c4a`; its pre-existing package-size correction
relative to the older oracle commit is not part of this experiment. The referee/clock logic is
unchanged, and `numba_core.py` is byte-for-byte the reviewed prototype.

## Draw semantics

Legal moves are generated before terminal checks so checkmate/stalemate have the same precedence
as python-chess. Insufficient material uses the same standard-chess material/color criteria.
At 100 halfmoves a live position is drawn. At 99 halfmoves, a claim exists only if at least one
legal non-zeroing move leaves the opponent with a legal move (not checkmate or stalemate).

Repetition uses exact 67-element snapshots, not probabilistic hashes or a transposition table:
piece placement, side, castling rights, and en-passant square only when a legal EP capture exists.
Each recursive ply stores its current key. The implementation checks both the third current
occurrence and a legal next move that reaches a position seen twice. The champion's seven-ply
claim guard is preserved. A FEN root starts with no history. Public Board APIs can replay an
explicit supplied move stack for differential tests; no pre-root game history is invented.

Keeping the complete available prefix instead of replaying only back to the last irreversible
move is safe: pawn progress/promotion, material loss, and lost castling rights cannot be reversed
to reproduce an earlier exact key. Legal EP availability is normalized identically. Hash
collisions cannot create false draw claims because comparison is element-by-element.

## Execution and safety

All recursive chess operations remain JIT-compiled. Python parses the root, computes the unchanged
time allocation, runs iterative deepening, converts the result, and supplies a legal fallback.
The portable deadline check briefly enters Python only to read a high-resolution monotonic clock
every 64 visited nodes. It uses a two-millisecond margin inside the allocated hard budget.
An abort flag unwinds every make/unmake before returning the last fully completed iteration.
The fixed 512-ply storage guard aborts rather than changing a score; fixed-depth callers receive
an error if the guard is reached. No disk JIT cache or compiled binary is shipped.

Import warmup exercises both fixed and timed calls with the same concrete scalar/array signatures
used in gameplay. Tests/initialization diagnostics verify no new dispatcher signature appears in
the first timed move. At an already-drawn root with legal moves, timed search returns the same
legal fallback directly instead of the Python oracle's futile deadline loop; fixed-depth terminal
scores are unchanged.

## Validation and benchmarks

The fixed-depth gate uses smoke + development-64 + 500 deterministic reachable positions
(seed 20260904):
depths 1 and 2 everywhere, and depth 3 on smoke/development (1,204 comparisons). All score/move
differences must be investigated before running timed games. Draw, mate, qsearch tactical,
restoration, and timeout tests supplement this gate.

Timed diagnostics and paired games use only development suites. Games call the unmodified
referee and sandbox; stderr telemetry records depth, normal/qnodes, move time, and engine errors.
Independent games may run in two workers; each engine's code is single-threaded. Results are
local-machine measurements, not a claim about competition hardware or Elo.

Promotion C is not opened or used by this experiment. Champion is never modified.

## Recorded correctness and initialization evidence

- 568 positions, 1,204 fixed-depth comparisons: **zero score mismatches and zero move differences**.
- Normal nodes, qnodes, beta cutoffs, killer searches, and history-ordered counters also match in
  every fixed-depth comparison. No unexplained equal-score move changes were present.
- Differential core tests additionally cover smoke/development and 2,048 deterministic random
  positions, comparing every legal move and make/unmake state against python-chess. Castling on
  both sides, en passant, all underpromotions, double pawn moves, pins, discovered/double check,
  checkmate, and stalemate passed. Differential perft totals matched: start depth 4 = 197,281;
  Kiwipete depth 3 = 97,862; endgame depth 3 = 2,812.
- Search tests cover fifty-move claims at 99/100 halfmoves, terminal precedence, immediate and
  next-move repetition claims, absent pre-root history, legal-EP key normalization, mate/draw
  scores, qsearch tactics, and restoration after deadline aborts.
- Cold total agent imports: 15.461, 16.336, and 15.619 seconds. All three first-gameplay signature
  audits showed no additional compilation. Peak working sets: 214,274,048, 214,511,616, and
  216,047,616 bytes (approximately 204–206 MiB).
- Ruff formatting/lint, strict mypy (29 source files), and 53 selected tests passed. The only
  excluded test module is the unchanged sealed-promotion-data validator, to avoid opening C.
- The archive contains exactly `agent.py`, `engine.py`, `numba_core.py`, and `numba_engine.py`:
  14,138 compressed bytes and 56,235 uncompressed bytes, below the exact 50,000,000-byte cap.
  An extracted-archive smoke move passed with no engine error.

Raw evidence is in `benchmarks/results/numba-search-v1-parity.json`, `numba-search-v1-init.json`,
and `numba-search-v1-package.json`.

## Development-fast-16 evidence (5,000 ms + 100 ms, 120 plies)

Paired result: **19 W / 12 D / 1 L**, score **78.125%**. All 32 games completed; zero referee
failures and zero engine-error fallbacks on either side. Wall time including cold initialization
was 409.86 seconds with two independent game workers.

Game telemetry: mean completed depth 4.276 (Numba) versus 2.489 (Python); Numba searched
3,194,119 normal nodes and 18,753,781 qnodes over 973 moves. Python searched 136,453 normal nodes
and 1,191,677 qnodes over 964 moves.

Root diagnostics at 5,000 ms remaining:

| Metric | Numba | Python champion |
|---|---:|---:|
| Normal nodes/s | 12,581 | 588 |
| Qnodes/s | 106,126 | 6,618 |
| Total nodes/s | 118,707 | 7,206 |
| Mean completed depth | 3.813 | 1.938 |
| Mean wall ms/move | 180.08 | 189.82 |

The longest measured Numba root call was 190.86 ms, inside the 192 ms allocated hard budget.

## Development-64 evidence (5,000 ms + 100 ms, 120 plies)

Paired result: **77 W / 42 D / 9 L**, score **76.5625%**. All 128 games completed; zero referee
failures and zero engine-error fallbacks on either side. Wall time including initialization was
1,575.74 seconds with two independent workers.

Mean completed depth was 4.358 (Numba) versus 2.573 (Python), a difference of +1.785. Numba
searched 12,172,215 normal nodes and 68,908,135 qnodes over 3,517 moves. Python searched 533,614
normal nodes and 4,367,635 qnodes over 3,479 moves.

Root diagnostics at 5,000 ms remaining:

| Metric | Numba | Python champion |
|---|---:|---:|
| Normal nodes/s | 20,514 | 882 |
| Qnodes/s | 123,250 | 8,101 |
| Total nodes/s | 143,764 | 8,983 |
| Mean completed depth | 4.078 | 2.266 |
| Mean wall ms/move | 181.88 | 179.63 |

This is approximately a 16.0x measured total-node throughput improvement. These development
results justified advancing to the separately reported competition-clock test; they did not
trigger any parameter or heuristic changes.

## Competition-clock development-fast-16 (120,000 ms + 500 ms, 300 plies)

Paired result: **21 W / 11 D / 0 L**, score **82.8125%**. All 32 games completed; zero referee
failures and zero engine-error fallbacks on either side. Wall time including initialization was
2,611.62 seconds (43.53 minutes) with two independent workers.

| Game telemetry | Numba | Python champion |
|---|---:|---:|
| Moves | 1,106 | 1,095 |
| Mean completed depth | 5.778 | 3.939 |
| Normal nodes | 40,224,020 | 1,687,663 |
| Qnodes | 213,706,803 | 10,910,575 |
| Mean reported search ms/move | 2,001.69 | 2,037.86 |
| Total reported search seconds | 2,213.87 | 2,231.46 |

The mean completed-depth advantage was **+1.839 plies**. Search-time telemetry excludes import
and outer protocol overhead; run wall time includes them. For comparison, mean reported search
ms/move in the shorter matches was 143.08 / 147.85 (fast-16) and 143.82 / 148.77 (development-64),
Numba / Python respectively.

All three match runs used frozen candidate code `b4792736594965a33c187717a0c61f06250398a0` and
exact oracle `6686f7ca8412776247149512a28fe7ec18f034f9`. The final reporting commit changes only
this document and recorded results. The development suites overlap, so their scores must not be
pooled as independent evidence or interpreted as a formal Elo estimate.

## Reproduction and retained evidence

Run from the experiment checkout with an exact, separate oracle worktree:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m unittest tests.test_engine tests.test_numba_core tests.test_numba_search tests.test_build_position_suite
uv run python -m tools.validate_numba_search
uv run python -m tools.benchmark_numba_search --suite development-fast-16
uv run python -m tools.benchmark_numba_search --suite development-64
uv run python -m tools.run_numba_matches --suite development-fast-16 --opponent ORACLE_WORKTREE --label development-fast-16
uv run python -m tools.run_numba_matches --suite development-64 --opponent ORACLE_WORKTREE --label development-64
uv run python -m tools.run_numba_matches --suite development-fast-16 --opponent ORACLE_WORKTREE --label competition-clock-fast-16 --base-ms 120000 --increment-ms 500 --ply-cap 300 --workers 2
uv run python -m tools.measure_numba_init
uv run python -m harness.package
uv run python -m tools.validate_numba_package
```

Mypy uses the configured source-file list (the baseline directories contain separate modules
named `agent`). The sealed-promotion validator is deliberately excluded from test invocation.

Retained JSON files under `benchmarks/results/`, all prefixed `numba-search-v1-`:

- `parity.json`: all 1,204 comparisons, scores, moves, counters and timings.
- `init.json`: three fresh imports, memory and post-warmup signature audits.
- `package.json`: exact archive manifest, sizes, hashes and extracted-package smoke result.
- `development-fast-16-diagnostics.json` and `development-64-diagnostics.json`: timed roots.
- `development-fast-16.json`, `development-64.json`, and `competition-clock-fast-16.json`:
  paired results, per-move telemetry, PGNs, clock settings and both exact code revisions.

No promotion suite or holdout was used. No new heuristic was added, no champion ref was moved,
and the experiment is not merged.
