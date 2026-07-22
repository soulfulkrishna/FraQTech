from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "requirements.txt", "requirements-qpu.txt", "environment.yml",
    "AI_USE_DISCLOSURE.md", "qbraid_skill/SKILL.md",
    "scripts/smoke_test.py", "scripts/run_phase3_classical.py",
    "scripts/run_phase3_qrc_sim.py", "scripts/run_phase3_noise.py",
    "scripts/run_phase3_ibm.py", "scripts/run_phase3_mnist.py",
    "scripts/run_phase3_shot_sweep.py", "scripts/run_phase3_ablations.py",
    "scripts/assemble_final_pdf.py", "writeup/main.tex", "writeup/references.bib",
    "writeup/main.pdf",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    errors, warnings = [], []
    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"missing: {rel}")
    if not compileall.compile_dir(ROOT / "src", quiet=1) or not compileall.compile_dir(ROOT / "scripts", quiet=1):
        errors.append("Python compilation failed")
    dataset = ROOT / "data/processed/finance_vix_vol_h1.csv"
    if dataset.exists():
        manifest = ROOT / "data/manifests/finance_vix_vol_h1.json"
        if not manifest.exists():
            warnings.append("processed dataset exists but manifest is missing")
        else:
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info.get("processed_sha256") != sha256(dataset):
                errors.append("dataset checksum differs from manifest")
    else:
        warnings.append("processed VIX dataset not yet built")
    raw_results = list((ROOT / "results/raw").glob("*.json"))
    if args.require_results and not raw_results:
        errors.append("no Phase 3 result JSON files")
    elif not raw_results:
        warnings.append("remaining Phase 3 results are intentionally absent")
    if "YOUR_GITHUB_REPO" in (ROOT / "README.md").read_text(encoding="utf-8"):
        warnings.append("replace the Launch-on-qBraid GitHub placeholder after publishing the repo")
    status = {"ok": not errors, "errors": errors, "warnings": warnings, "result_files": len(raw_results)}
    print(json.dumps(status, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
