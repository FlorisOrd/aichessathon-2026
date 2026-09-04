"""Inspect the submission archive and smoke-test only its extracted source files."""

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    archive_path = Path("submission.zip")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        size = sum(member.file_size for member in members)
        if "agent.py" not in archive.namelist() or size > 50_000_000:
            raise SystemExit("Invalid package root or size")
        if any("/" in member.filename or not member.filename.endswith(".py") for member in members):
            raise SystemExit("Expected root-level Python source files only")
        manifest = [
            {
                "path": member.filename,
                "bytes": member.file_size,
                "sha256": hashlib.sha256(archive.read(member)).hexdigest(),
            }
            for member in members
        ]
        with tempfile.TemporaryDirectory(prefix="numba-package-") as temporary:
            archive.extractall(temporary)
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import agent,chess; b=chess.Board(); m=agent.get_move(b.fen(),1000); "
                    "assert chess.Move.from_uci(m) in b.legal_moves; print('PACKAGE_OK')",
                ],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            if "PACKAGE_OK" not in process.stdout or "engine_error=" in process.stdout:
                raise SystemExit("Packaged agent smoke check failed")
    result = {
        "compressed_bytes": archive_path.stat().st_size,
        "uncompressed_bytes": size,
        "cap_bytes": 50_000_000,
        "manifest": manifest,
        "packaged_smoke_passed": True,
        "log": process.stdout,
    }
    Path("benchmarks/results/numba-search-v1-package.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
