"""Run reproducible, paired-FEN experiments through the official referee."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import chess

from harness.referee import FAILED_TERMINATIONS, Outcome, Result, play_match
from harness.sandbox import local

SCHEMA = "aichessathon.arena-suite"
SCHEMA_VERSION = 1
DEFAULT_BASE_MS = 5_000
DEFAULT_INCREMENT_MS = 100
DEFAULT_PLY_CAP = 120
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIONS = REPOSITORY_ROOT / "benchmarks" / "positions" / "smoke.fens"
AGENT_FAILURE_EXIT_CODE = 2

CandidateColor = Literal["white", "black"]
CandidateResult = Literal["win", "draw", "loss", "void"]
FailedAgent = Literal["candidate", "opponent"]


class ArenaSuiteError(Exception):
    """An invalid configuration or experiment infrastructure failure."""


@dataclass(frozen=True)
class Position:
    index: int
    source_line: int
    fen: str


@dataclass(frozen=True)
class Configuration:
    candidate: Path
    opponent: Path
    position_suite: Path
    seed: int
    shuffle: bool
    repeats: int
    base_ms: int
    increment_ms: int
    ply_cap: int
    json_output: Path | None
    pgn_output: Path | None
    fail_fast: bool


@dataclass(frozen=True)
class AgentMetadata:
    path: str
    git_commit: str | None


@dataclass(frozen=True)
class SuiteMetadata:
    path: str
    positions: int


@dataclass(frozen=True)
class ExperimentConfiguration:
    seed: int
    shuffle: bool
    repeats: int
    base_ms: int
    increment_ms: int
    ply_cap: int
    fail_fast: bool


@dataclass(frozen=True)
class GameRecord:
    game_number: int
    repeat: int
    position_index: int
    position_source_line: int
    fen: str
    candidate_color: CandidateColor
    candidate_result: CandidateResult
    referee_result: Result
    termination: str
    failed_agents: tuple[FailedAgent, ...]


@dataclass(frozen=True)
class ExperimentResult:
    schema: str
    schema_version: int
    timestamp_utc: str
    candidate: AgentMetadata
    opponent: AgentMetadata
    position_suite: SuiteMetadata
    configuration: ExperimentConfiguration
    planned_games: int
    total_games: int
    wins: int
    draws: int
    losses: int
    voids: int
    score: float
    score_games: int
    termination_counts: dict[str, int]
    agent_failure_games: int
    elapsed_seconds: float
    pgn_output: str | None
    games: list[GameRecord]


@dataclass(frozen=True)
class RunArtifacts:
    result: ExperimentResult
    pgns: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two local agents over paired starting positions using the official referee. "
            "Agent failures are recorded and the full schedule finishes by default; the final "
            "exit status is 2 when any agent failed."
        )
    )
    parser.add_argument("--agent", type=Path, required=True, help="Candidate agent directory.")
    parser.add_argument("--opponent", type=Path, required=True, help="Opponent agent directory.")
    parser.add_argument(
        "--positions",
        type=Path,
        default=DEFAULT_POSITIONS,
        help=f"FEN suite (default: {DEFAULT_POSITIONS}).",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Complete suite repetitions.")
    parser.add_argument("--base-ms", type=int, default=DEFAULT_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=DEFAULT_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=DEFAULT_PLY_CAP)
    parser.add_argument("--seed", type=int, default=0, help="Seed recorded with every run.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Deterministically shuffle position order using --seed.",
    )
    parser.add_argument("--json-output", type=Path, help="Write the compact run record as JSON.")
    parser.add_argument(
        "--pgn-output",
        type=Path,
        help="Write all game PGNs to a separate file in scheduled order.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first agent failure instead of completing the schedule.",
    )
    return parser


def configuration_from(arguments: argparse.Namespace) -> Configuration:
    candidate = _agent_directory(cast(Path, arguments.agent), "candidate")
    opponent = _agent_directory(cast(Path, arguments.opponent), "opponent")
    position_suite = cast(Path, arguments.positions).resolve()
    repeats = cast(int, arguments.repeats)
    base_ms = cast(int, arguments.base_ms)
    increment_ms = cast(int, arguments.increment_ms)
    ply_cap = cast(int, arguments.ply_cap)
    json_output = _resolved_optional_path(cast(Path | None, arguments.json_output))
    pgn_output = _resolved_optional_path(cast(Path | None, arguments.pgn_output))

    if repeats < 1:
        raise ArenaSuiteError("--repeats must be at least 1")
    if base_ms < 1:
        raise ArenaSuiteError("--base-ms must be at least 1")
    if increment_ms < 0:
        raise ArenaSuiteError("--increment-ms cannot be negative")
    if ply_cap < 1:
        raise ArenaSuiteError("--ply-cap must be at least 1")
    if json_output is not None and json_output == pgn_output:
        raise ArenaSuiteError("--json-output and --pgn-output must be different files")

    return Configuration(
        candidate=candidate,
        opponent=opponent,
        position_suite=position_suite,
        seed=cast(int, arguments.seed),
        shuffle=cast(bool, arguments.shuffle),
        repeats=repeats,
        base_ms=base_ms,
        increment_ms=increment_ms,
        ply_cap=ply_cap,
        json_output=json_output,
        pgn_output=pgn_output,
        fail_fast=cast(bool, arguments.fail_fast),
    )


def load_positions(path: Path) -> list[Position]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ArenaSuiteError(f"cannot read position suite {path}: {error}") from error

    positions: list[Position] = []
    for source_line, raw_line in enumerate(lines, start=1):
        fen = raw_line.strip()
        if not fen or fen.startswith("#"):
            continue
        try:
            board = chess.Board(fen)
        except ValueError as error:
            message = f"invalid FEN in {path} at line {source_line}: {error}"
            raise ArenaSuiteError(message) from error
        if not board.is_valid():
            raise ArenaSuiteError(
                f"invalid FEN in {path} at line {source_line}: "
                f"illegal position (python-chess status {board.status()})"
            )
        positions.append(Position(index=len(positions) + 1, source_line=source_line, fen=fen))

    if not positions:
        raise ArenaSuiteError(f"position suite {path} contains no FEN positions")
    return positions


def run_experiment(configuration: Configuration, positions: list[Position]) -> RunArtifacts:
    ordered_positions = list(positions)
    if configuration.shuffle:
        random.Random(configuration.seed).shuffle(ordered_positions)

    planned_games = len(ordered_positions) * 2 * configuration.repeats
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    started_at = time.monotonic()
    games: list[GameRecord] = []
    pgns: list[str] = []
    stop = False

    for repeat in range(1, configuration.repeats + 1):
        for position in ordered_positions:
            for candidate_plays_white in (True, False):
                game_number = len(games) + 1
                white_path, black_path = (
                    (configuration.candidate, configuration.opponent)
                    if candidate_plays_white
                    else (configuration.opponent, configuration.candidate)
                )
                try:
                    outcome = play_match(
                        local(white_path),
                        local(black_path),
                        configuration.base_ms,
                        configuration.increment_ms,
                        ply_cap=configuration.ply_cap,
                        start_fen=position.fen,
                    )
                except Exception as error:
                    raise ArenaSuiteError(
                        f"infrastructure error while running game {game_number}: {error}"
                    ) from error

                candidate_color: CandidateColor = "white" if candidate_plays_white else "black"
                failed_agents = _failed_agents(outcome, candidate_color)
                record = GameRecord(
                    game_number=game_number,
                    repeat=repeat,
                    position_index=position.index,
                    position_source_line=position.source_line,
                    fen=position.fen,
                    candidate_color=candidate_color,
                    candidate_result=_candidate_result(outcome.result, candidate_color),
                    referee_result=outcome.result,
                    termination=outcome.termination,
                    failed_agents=failed_agents,
                )
                games.append(record)
                pgns.append(outcome.pgn)
                _print_game(record, planned_games)

                if failed_agents and configuration.fail_fast:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    elapsed_seconds = time.monotonic() - started_at
    result = _build_result(
        configuration,
        positions,
        games,
        planned_games,
        timestamp,
        elapsed_seconds,
    )
    return RunArtifacts(result=result, pgns=pgns)


def _build_result(
    configuration: Configuration,
    positions: list[Position],
    games: list[GameRecord],
    planned_games: int,
    timestamp: str,
    elapsed_seconds: float,
) -> ExperimentResult:
    wins = sum(game.candidate_result == "win" for game in games)
    draws = sum(game.candidate_result == "draw" for game in games)
    losses = sum(game.candidate_result == "loss" for game in games)
    voids = sum(game.candidate_result == "void" for game in games)
    score_games = wins + draws + losses
    score = (wins + draws / 2) / score_games if score_games else 0.0
    terminations: dict[str, int] = {}
    for game in games:
        terminations[game.termination] = terminations.get(game.termination, 0) + 1

    return ExperimentResult(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        timestamp_utc=timestamp,
        candidate=AgentMetadata(
            path=str(configuration.candidate),
            git_commit=_git_commit(configuration.candidate),
        ),
        opponent=AgentMetadata(
            path=str(configuration.opponent),
            git_commit=_git_commit(configuration.opponent),
        ),
        position_suite=SuiteMetadata(
            path=str(configuration.position_suite),
            positions=len(positions),
        ),
        configuration=ExperimentConfiguration(
            seed=configuration.seed,
            shuffle=configuration.shuffle,
            repeats=configuration.repeats,
            base_ms=configuration.base_ms,
            increment_ms=configuration.increment_ms,
            ply_cap=configuration.ply_cap,
            fail_fast=configuration.fail_fast,
        ),
        planned_games=planned_games,
        total_games=len(games),
        wins=wins,
        draws=draws,
        losses=losses,
        voids=voids,
        score=score,
        score_games=score_games,
        termination_counts=dict(sorted(terminations.items())),
        agent_failure_games=sum(bool(game.failed_agents) for game in games),
        elapsed_seconds=round(elapsed_seconds, 6),
        pgn_output=str(configuration.pgn_output) if configuration.pgn_output else None,
        games=games,
    )


def _candidate_result(result: Result, candidate_color: CandidateColor) -> CandidateResult:
    if result == "draw":
        return "draw"
    if result == "void":
        return "void"
    candidate_won = result == candidate_color
    return "win" if candidate_won else "loss"


def _failed_agents(outcome: Outcome, candidate_color: CandidateColor) -> tuple[FailedAgent, ...]:
    if outcome.termination not in FAILED_TERMINATIONS:
        return ()
    if outcome.termination == "both_failed":
        return ("candidate", "opponent")
    if outcome.result not in {"white", "black"}:
        return ()
    failed_color = "black" if outcome.result == "white" else "white"
    return ("candidate",) if failed_color == candidate_color else ("opponent",)


def _print_game(game: GameRecord, planned_games: int) -> None:
    failure = ""
    if game.failed_agents:
        failure = " | AGENT FAILURE: " + ", ".join(game.failed_agents)
    print(
        f"game {game.game_number}/{planned_games} | repeat {game.repeat} | "
        f"position {game.position_index} | candidate {game.candidate_color} | "
        f"{game.candidate_result} | {game.referee_result} by {game.termination}{failure}"
    )


def print_summary(result: ExperimentResult) -> None:
    print("\nsummary")
    print(
        f"+{result.wins} ={result.draws} -{result.losses}, "
        f"voids {result.voids}, score {result.score:.1%}"
    )
    print(f"games {result.total_games}/{result.planned_games}")
    print(
        "terminations: "
        + ", ".join(f"{name} {count}" for name, count in result.termination_counts.items())
    )
    print(f"agent failure games: {result.agent_failure_games}")
    print(f"elapsed: {result.elapsed_seconds:.3f}s")


def write_outputs(configuration: Configuration, artifacts: RunArtifacts) -> None:
    if configuration.json_output is not None:
        _write_text(
            configuration.json_output,
            json.dumps(asdict(artifacts.result), indent=2, sort_keys=True) + "\n",
        )
        print(f"json written to {configuration.json_output}")
    if configuration.pgn_output is not None:
        pgn_text = "\n\n".join(pgn.rstrip() for pgn in artifacts.pgns) + "\n"
        _write_text(configuration.pgn_output, pgn_text)
        print(f"pgn written to {configuration.pgn_output}")


def _agent_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ArenaSuiteError(f"{label} agent directory does not exist: {resolved}")
    if not (resolved / "agent.py").is_file():
        raise ArenaSuiteError(f"{label} agent directory has no agent.py: {resolved}")
    return resolved


def _resolved_optional_path(path: Path | None) -> Path | None:
    return path.resolve() if path is not None else None


def _write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ArenaSuiteError(f"cannot write {path}: {error}") from error


def _git_commit(directory: Path) -> str | None:
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def main() -> None:
    parser = build_parser()
    try:
        configuration = configuration_from(parser.parse_args())
        positions = load_positions(configuration.position_suite)
        artifacts = run_experiment(configuration, positions)
        print_summary(artifacts.result)
        write_outputs(configuration, artifacts)
    except ArenaSuiteError as error:
        parser.exit(1, f"error: {error}\n")

    if artifacts.result.agent_failure_games:
        raise SystemExit(AGENT_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
