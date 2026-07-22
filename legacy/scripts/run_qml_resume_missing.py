import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["narma10", "narma20", "mackey_glass", "lorenz63", "santafe"]
DEFAULT_MODELS = ["qnn", "qnn_ising"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_csv(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def result_path(root: Path, dataset: str, model: str, seed: int) -> Path:
    return root / "results" / "raw_logs" / f"{dataset}_{model}_seed{seed}.json"


def run_and_log(cmd, log_path: Path) -> int:
    print("\nRUN:", " ".join(cmd), flush=True)

    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\nRUN: " + " ".join(cmd) + "\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if output JSON exists.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "results" / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_path = root / "results" / "qml_resume_missing.log"
    plan_path = root / "results" / "qml_resume_plan.csv"

    datasets = parse_csv(args.datasets)
    models = parse_csv(args.models)
    seeds = [int(x) for x in parse_csv(args.seeds)]

    jobs = []
    skipped = []

    for dataset in datasets:
        for model in models:
            for seed in seeds:
                out = result_path(root, dataset, model, seed)
                if out.exists() and not args.force:
                    skipped.append((dataset, model, seed, out.name))
                else:
                    jobs.append((dataset, model, seed, out.name))

    with open(plan_path, "w", encoding="utf-8") as f:
        f.write("status,dataset,model,seed,expected_output\n")
        for dataset, model, seed, name in skipped:
            f.write(f"skip,{dataset},{model},{seed},{name}\n")
        for dataset, model, seed, name in jobs:
            f.write(f"run,{dataset},{model},{seed},{name}\n")

    print("\nQML resume plan")
    print("===============")
    print(f"Python: {sys.executable}")
    print(f"Root: {root}")
    print("Experiment script: scripts\\run_qml_experiment.py")
    print("Model argument flag: --qml-config")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Total planned: {len(jobs) + len(skipped)}")
    print(f"Already done: {len(skipped)}")
    print(f"Remaining to run: {len(jobs)}")
    print(f"Plan CSV: {plan_path}")
    print(f"Log file: {log_path}")

    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n\nQML resume plan\n")
        log.write("================\n")
        log.write(f"Python: {sys.executable}\n")
        log.write(f"Root: {root}\n")
        log.write("Experiment script: scripts\\run_qml_experiment.py\n")
        log.write("Model argument flag: --qml-config\n")
        log.write(f"Datasets: {datasets}\n")
        log.write(f"Models: {models}\n")
        log.write(f"Seeds: {seeds}\n")
        log.write(f"Total planned: {len(jobs) + len(skipped)}\n")
        log.write(f"Already done: {len(skipped)}\n")
        log.write(f"Remaining to run: {len(jobs)}\n")

    for i, (dataset, model, seed, expected_name) in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] dataset={dataset} model={model} seed={seed}")

        cmd = [
            sys.executable,
            "scripts\\run_qml_experiment.py",
            "--dataset",
            dataset,
            "--qml-config",
            model,
            "--seed",
            str(seed),
        ]

        if args.no_energy:
            cmd.append("--no-energy")

        return_code = run_and_log(cmd, log_path)

        if return_code != 0:
            print(f"\nFAILED: dataset={dataset} model={model} seed={seed} return_code={return_code}")
            if not args.continue_on_error:
                raise SystemExit(return_code)

    print("\nQML resume complete.")
    print(f"Skipped existing: {len(skipped)}")
    print(f"Newly attempted: {len(jobs)}")
    print(f"Plan CSV: {plan_path}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()