# AI Chessathon 2026 Agent

**A deterministic chess agent with time-managed search and reproducible evaluation.**

This repository is an active competition-development project built on the upstream [AI Chessathon starter](https://github.com/advitrocks9/aichessathon-starter). The current candidate replaces the starter's random mover with a deterministic search engine and adds controlled evaluation tooling. No competition ranking, Elo estimate or uncommitted benchmark result is claimed here.

## Upstream starter and repository-specific work

The upstream starter provides the platform-compatible `get_move` contract, process sandbox, referee and clock handling, local play and arena harnesses, baseline agents, packaging flow and Make targets. Those components retain their original attribution and [MIT licence](LICENSE).

Work developed in this repository includes:

- `agent.py`, which calls the candidate engine and returns a deterministic legal fallback if search fails or time is too low.
- `engine.py`, implementing iterative-deepening negamax, alpha-beta pruning, deterministic move ordering, quiescence search and a tapered material/piece-square evaluation.
- Conservative soft and hard time budgets that preserve a clock reserve and return the last completed search iteration.
- Reproducible paired-position experiments through `tools.arena_suite`, including reversed colours, fixed seeds, explicit time controls, position-suite hashes and optional JSON or PGN records.
- Fixed development and holdout FEN suites with documented provenance and usage policy.

The upstream `harness/` referee remains authoritative for local match rules and legality.

## Agent contract

The platform imports `agent.py` and calls:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The current implementation parses the FEN with `python-chess`, searches from the side to move and returns a legal move in UCI notation.

## Set up and play

```bash
git clone https://github.com/FlorisOrd/aichessathon-2026.git
cd aichessathon-2026
make setup
make play
```

Useful Make targets:

```bash
make play
make arena
make zip
make gate
```

- `make play` runs one game against the greedy baseline.
- `make arena` runs a 20-game local match against the greedy baseline.
- `make zip` builds `submission.zip`.
- `make gate` runs Ruff, mypy and a short harness match against the random baseline.

Start from a specific position:

```bash
make play FEN="<fen>"
```

Run a single game with explicit sides and save the PGN:

```bash
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
```

Run a larger local match:

```bash
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

CI runs the quality gate on Ubuntu and a short harness match on macOS and Windows.

## Reproducible evaluation

`tools.arena_suite` compares two local agent directories across a FEN suite. Each position is played twice with colours reversed to control for colour and starting-position effects.

Fast candidate-versus-baseline smoke run from PowerShell:

```powershell
uv run python -m tools.arena_suite --agent . --opponent baselines/greedy `
  --json-output benchmarks/results/smoke.json --pgn-output benchmarks/results/smoke.pgn
```

Candidate-versus-champion example:

```powershell
uv run python -m tools.arena_suite --agent C:\worktrees\candidate `
  --opponent C:\worktrees\champion --repeats 10 --shuffle --seed 2026 `
  --base-ms 120000 --increment-ms 500 --ply-cap 300 `
  --json-output benchmarks/results/competition.json
```

The runner normally completes the planned schedule before returning exit status `2` when an agent fails. Add `--fail-fast` to stop after the first crash, illegal move, flag, initialization failure or `both_failed`.

See [fixed benchmark suites](docs/BENCHMARK_SUITES.md) for provenance, regeneration and the separation between development and holdout positions. The general starter guidance remains available in [Where the strength comes from](docs/IDEAS.md).

## Current status and limitations

- Search order and evaluation are deterministic; wall-clock timing can affect the last completed depth, but playing strength is not asserted without committed, traceable experiment records.
- The engine has no transposition table or external chess engine integration.
- A root FEN contains a halfmove clock but not the full earlier repetition history; the engine does not invent unavailable game history.
- Generated benchmark JSON and PGN files are local artefacts unless deliberately committed with their provenance and reproduction command.
- Platform rules and upload validation remain authoritative. Consult the current [AI Chessathon documentation](https://aichessathon.com/docs) before submission.

## My role and development approach

I owned the architecture, experiment design, task decomposition, testing, benchmark decisions and coordination of coding agents. Claude-assisted changes were evaluated through tests and controlled comparisons rather than accepted solely because they appeared plausible.
