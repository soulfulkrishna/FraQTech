from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="FraQTech")
    parser.add_argument("--challenge", default="DynamicSystemsForecasting")
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, "scripts/validate_submission.py"] + (["--require-results"] if args.require_results else []),
        cwd=ROOT, check=True,
    )
    output = ROOT.parent / f"{args.team}_{args.challenge}_Phase3.zip"
    excludes = {".git", "__pycache__", ".pytest_cache", "feature_cache"}
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / f"{args.team}_{args.challenge}_Phase3"
        shutil.copytree(
            ROOT, stage,
            ignore=shutil.ignore_patterns(
                "*.pyc", "__pycache__", ".git", ".pytest_cache", "feature_cache",
                "*.aux", "*.log", "*.out"
            ),
        )
        manifest_lines = []
        for file_path in sorted(x for x in stage.rglob("*") if x.is_file()):
            if file_path.name == "SUBMISSION_MANIFEST_SHA256.txt":
                continue
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            manifest_lines.append(f"{digest}  {file_path.relative_to(stage).as_posix()}")
        (stage / "SUBMISSION_MANIFEST_SHA256.txt").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        if output.exists():
            output.unlink()
        shutil.make_archive(str(output.with_suffix("")), "zip", root_dir=stage.parent, base_dir=stage.name)
    print(f"Created: {output}")


if __name__ == "__main__":
    main()
