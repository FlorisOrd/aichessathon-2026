"""Measure fresh-process total agent import, then verify no gameplay JIT specialization."""

import argparse
import ctypes
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType


def peak_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("faults", ctypes.c_ulong),
            ("peak_working_set", ctypes.c_size_t),
            ("working_set", ctypes.c_size_t),
            ("peak_paged", ctypes.c_size_t),
            ("paged", ctypes.c_size_t),
            ("peak_nonpaged", ctypes.c_size_t),
            ("nonpaged", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t),
            ("peak_pagefile", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not query(process(), ctypes.byref(counters), counters.size):
        return None
    return int(counters.peak_working_set)


def signatures(module: ModuleType) -> dict[str, str]:
    return {
        name: str(value.signatures)
        for name, value in vars(module).items()
        if hasattr(value, "signatures")
    }


def child() -> None:
    started = time.perf_counter()
    agent = importlib.import_module("agent")
    elapsed = time.perf_counter() - started
    modules = [importlib.import_module(name) for name in ("numba_core", "numba_engine")]
    before = [signatures(module) for module in modules]
    cases = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/P6k/8/8/8/8/8/4K3 w - - 0 1",
        "k3r3/8/8/8/8/8/8/4K3 w - - 99 1",
    )
    moves = [agent.get_move(fen, 1000) for fen in cases]
    after = [signatures(module) for module in modules]
    print(
        "INIT_RESULT "
        + json.dumps(
            {
                "cold_import_seconds": elapsed,
                "no_new_signatures": before == after,
                "moves": moves,
                "peak_working_set_bytes": peak_memory_bytes(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child()
        return
    records = []
    for _ in range(3):
        process = subprocess.run(
            [sys.executable, "-m", "tools.measure_numba_init", "--child"],
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        line = next(line for line in process.stdout.splitlines() if line.startswith("INIT_RESULT "))
        records.append(json.loads(line.removeprefix("INIT_RESULT ")))
        print(records[-1], flush=True)
    output = Path("benchmarks/results/numba-search-v1-init.json")
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if any(
        not record["no_new_signatures"] or record["cold_import_seconds"] >= 60 for record in records
    ):
        raise SystemExit("init acceptance failed")


if __name__ == "__main__":
    main()
