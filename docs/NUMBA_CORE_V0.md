# Numba core v0

`numba_core.py` is a standalone correctness/performance prototype. It is not imported by
`agent.py` or `engine.py`, so it does not change the competition agent's move selection.

## Representation and scope

The core uses a fixed `int16[69]` NumPy array: 64 signed piece codes followed by side to move,
four castling-right bits, en-passant square (`-1` for none), halfmove clock, and fullmove number.
Moves are compact integers containing source, destination, promotion, en-passant, and castling
fields. Make/unmake uses a fixed nine-element undo record and restores the entire state exactly.

The Numba hot paths cover attack detection, check detection, pseudo-legal generation, legal
filtering, make/unmake, and differential perft. The supported rules are standard chess; Chess960
is intentionally outside this prototype. The active alpha-beta search, qsearch, evaluation, and
agent entry point are unchanged.

All conversion and checking code uses documented public python-chess APIs. python-chess is only
the oracle and benchmark comparator; the internal core itself imports only NumPy and Numba.

## Differential validation

The fixed corpus contains the four smoke positions, all 64 development positions, and 2,048
reachable positions from deterministic random legal playouts (seed `20260903`). For each of the
2,116 positions, validation compares the complete legal UCI move set, check status, piece placement,
side, castling rights, en-passant square, halfmove clock, and fullmove number. Every one of the
64,861 legal moves is then made in both implementations, the resulting state and check status are
compared, and the internal move is undone with exact restoration required.

Result: **zero mismatches**.

Explicit cases pass for white and black kingside/queenside castling, en passant, queen promotion
and every underpromotion, double pawn moves, pins, discovered check, double check, checkmate, and
stalemate.

Differential perft results:

| Position | Depth | Numba | python-chess | Expected |
|---|---:|---:|---:|---:|
| Initial position | 4 | 197,281 | 197,281 | 197,281 |
| Kiwipete | 3 | 97,862 | 97,862 | 97,862 |
| Endgame reference | 3 | 2,812 | 2,812 | 2,812 |

## Performance snapshot

Measured on Windows 11, Python 3.12.14, Numba 0.67.0, NumPy 2.5.2, and python-chess 1.11.2.
The cold combined compile/warmup time for the relevant functions was **11.869 seconds**.

| Operation | Numba | python-chess | Speedup |
|---|---:|---:|---:|
| Legal move generation | 246,774 positions/s | 18,254 positions/s | 13.52x |
| Legal moves emitted | 7,564,268 moves/s | 559,545 moves/s | 13.52x |
| Make/unmake round trips | 1,053,111/s | 147,063/s | 7.16x |

These are steady-state microbenchmarks on the same fixed 2,116-position corpus. They establish that
the representation is viable; they do not predict whole-engine speed until a later project ports
search and evaluation. Raw measurements and environment details are committed in
`benchmarks/numba-core-v0-results.json`.

Regenerate the validation and benchmark report with:

```powershell
uv run python -m tools.benchmark_numba_core
```
