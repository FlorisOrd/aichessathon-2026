# Numba search v1 speed port

Semantic oracle: exact killer/history `6686f7ca8412776247149512a28fe7ec18f034f9`.
Base release: `3227c4a93750033a227b48882f057a77f7c937ae`.
Core imported by cherry-picking reviewed prototype `6a89bccbf58bc1cb7aa7b007e8238849b820e93d`.

`engine.py` remains byte-for-byte the Python oracle. `numba_engine.py` ports its tapered integer
evaluation, promotion/MVV-LVA ordering, UCI tie order, two killers, capped depth-squared history,
negamax alpha-beta, captures/promotions qsearch, all check evasions, and mate-distance scores.
No TT, pruning heuristic, evaluator change, pondering, or new chess feature is included.

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

The fixed-depth gate uses smoke + development-64 + 500 deterministic reachable positions:
depths 1 and 2 everywhere, and depth 3 on smoke/development (1,204 comparisons). All score/move
differences must be investigated before running timed games. Draw, mate, qsearch tactical,
restoration, and timeout tests supplement this gate.

Timed diagnostics and paired games use only development suites. Games call the unmodified
referee and sandbox; stderr telemetry records depth, normal/qnodes, move time, and engine errors.
Independent games may run in two workers; each engine's code is single-threaded. Results are
local-machine measurements, not a claim about competition hardware or Elo.

Promotion C is not opened or used by this experiment. Champion is never modified.
