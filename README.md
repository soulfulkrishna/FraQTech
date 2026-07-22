# FraQTech - Hardware-Portable Quantum Reservoir Computing for VIX Forecasting

**Global Industry Challenge 2026 - Dynamic Systems Forecasting**  
**Track A: Financial Volatility Prediction**

This repository is the Phase 3 execution package for a gate-model quantum reservoir computer (QRC) applied to next-day VIX realized-volatility forecasting and high-volatility regime-transition prediction. The quantum reservoir is frozen: only a classical ridge, TCN, or logistic readout is trained.

> **Status:** all code, configurations, documentation, paper source, qBraid Skill, hardware adapters, and result-generation scripts are present. New 5/10/15-qubit, noise, QPU, and MNIST result files are intentionally not pre-filled.

<!-- Replace YOUR_GITHUB_REPO.git after publishing. -->
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/soulfulkrishna/FraQTech.git)

## Headline pilot evidence already available

The earlier simulator benchmark found the clearest QRC benefit on VIX:

| Model | VIX NRMSE (mean +/- std, 5 seeds) | Interpretation |
|---|---:|---|
| CV-QRC baseline + TCN | **0.4127 +/- 0.0062** | Best pilot VIX NRMSE |
| Classical TCN | 0.4167 +/- 0.0020 | Strongest classical neural baseline |
| Ridge-lag | 0.4534 | Efficient linear baseline |
| ESN | 0.4726 +/- 0.0043 | Classical reservoir baseline |
| Legacy GB-QRC + TCN | 1.1070 +/- 0.0342 | Negative baseline motivating the new design |

The Phase 3 code does **not** claim that the old GB-QRC is competitive. It replaces it with a validation-selected, hardware-portable temporal Ising QRC and tests whether any simulator benefit survives finite shots, realistic noise, and physical QPU execution.

## What is new in Phase 3

- Explicit use of the CSV `target` column; no implicit reconstruction from `value`.
- Training-only robust scaling and chronological train/validation/test separation.
- Future-shifted regime labels and explicit low-to-high transition targets.
- GARCH(1,1), HAR-RV, ESN, ridge-lag, persistence, and TCN baselines.
- Volatility-specific QLIKE, Mincer-Zarnowitz calibration, and Diebold-Mariano comparisons.
- Fixed temporal Ising QRC with repeated `RY/RZ` encoding, fixed `RX/RZ` fields, `RZZ` couplings, temporal multiplexing, and `Z/ZZ` observables.
- 5/10/15-qubit simulator scaling; architecture selection on validation only.
- Finite-shot, depolarizing, and amplitude-damping studies with Qiskit Aer.
- IBM Qiskit Runtime EstimatorV2 hardware execution and qBraid-managed non-IBM QPU execution.
- Common MNIST QRC expressivity benchmark.
- Prediction-level CSVs, resource metadata, circuit depth/gate/shot accounting, result aggregation, figures, and statistical comparison.

## Repository map

```text
configs/                 Experiment and hardware configurations
data/                    Public-data builder outputs and manifests
docs/                    Execution, IBM, platform, and submission guides
legacy/                  Unmodified prior benchmark code for provenance
notebooks/               qBraid quickstart notebook
qbraid_skill/            Agent-executable workflow instructions
results/                  Raw JSON, predictions, hardware metadata, figures
scripts/                  Reproducible command-line entry points
src/phase3/               New Phase 3 implementation
src/                      Prior reusable models and utilities
tests/                    Determinism and smoke tests
writeup/                  Five-page paper source and bibliography
```

## 1. qBraid setup

In qBraid Lab, open a terminal inside the cloned repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-qpu.txt
```

Run the no-credential smoke test:

```bash
python scripts/smoke_test.py
pytest -q
```

Expected: `status: PASS` and all tests pass.

## 2. Build the frozen public VIX dataset

```bash
python scripts/build_phase3_finance_data.py \
  --ticker '^VIX' \
  --start 2015-01-01 \
  --end 2026-01-01
```

Outputs:

- `data/processed/finance_vix_vol_h1.csv`
- `data/manifests/finance_vix_vol_h1.json`
- SHA-256 checksum and exact date range

For a frozen archival snapshot, commit the processed CSV and manifest to the final public repository. Judges should not depend on a future Yahoo response changing.

## 3. Fast local/qBraid verification

```bash
python scripts/reproduce_all.py --profile smoke --skip-data
```

This runs the core classical controls and a small 5-qubit exact-QRC test.

## 4. Classical benchmark

Five seeds:

```bash
for s in 0 1 2 3 4; do
  python scripts/run_phase3_classical.py --seed "$s" --profile full
done
```

Models: persistence, ridge-lag, HAR-RV, ESN, TCN, GARCH(1,1). Hyperparameters are selected on validation QLIKE; the test split is evaluated once.

## 5. Select the gate-QRC architecture

Start with the smoke search:

```bash
python scripts/select_gate_qrc_architecture.py \
  --profile smoke --seed 0 --device auto
```

Then run the full validation-only search on a qBraid GPU:

```bash
python scripts/select_gate_qrc_architecture.py \
  --profile full --seed 0 --device cuda \
  --max-train 600 --max-val 250
```

Copy the winning validation configuration into `configs/gate_qrc_5q.yaml`, record the selection CSV, and **freeze it before the final test sweep**.

## 6. 5/10/15-qubit simulator scaling

Initial bounded run:

```bash
python scripts/run_phase3_scaling.py \
  --qubits 5,10,15 --seeds 0,1,2 --readouts ridge,tcn \
  --device cuda --max-train 800 --max-val 250 --max-test 250
```

Final run after runtime is known:

```bash
python scripts/run_phase3_scaling.py \
  --qubits 5,10,15 --seeds 0,1,2 --readouts ridge,tcn \
  --device cuda --force
```

The 15-qubit exact statevector run is intentionally batch-limited and can be expensive. A complete 5/10-qubit result plus a documented bounded 15-qubit run is preferable to an unreproducible oversized sweep.

## 7. Ablations and regime-transition prediction

```bash
for s in 0 1 2; do
  python scripts/run_phase3_ablations.py --seed "$s" --device cuda
done

python scripts/run_phase3_regime.py --feature-source raw
python scripts/run_phase3_regime.py --feature-source esn
python scripts/run_phase3_regime.py --feature-source qrc --device cuda
```

Ablations include Z-only observables, no input re-uploading, line topology, and a matched-dimensional classical random-feature control.

## 8. Finite-shot and noise study

Pure shot-budget sweep under ideal sampling:

```bash
python scripts/run_phase3_shot_sweep.py \
  --shots 256,1024,4096 --seeds 0,1,2 --sample-windows 100
```

Noise sweep at the frozen 1024-shot budget:

```bash
for s in 0 1 2; do
  python scripts/run_phase3_noise.py \
    --seed "$s" --sample-windows 100 --shots 1024 \
    --conditions low_noise,medium_noise,high_noise
done
```

The ridge alpha is selected on validation-only exact features, then the readout is refit on train+validation and frozen across all shot/noise conditions. This isolates measurement and noise-induced feature degradation without test leakage.

## 9. MNIST common benchmark

```bash
python scripts/run_phase3_mnist.py --qubits 5  --seed 0 --device cuda
python scripts/run_phase3_mnist.py --qubits 10 --seed 0 --device cuda
python scripts/run_phase3_mnist.py --qubits 15 --seed 0 --device cuda
```

Default: fixed 2,000-train/500-test subset, 8x8 images treated as an eight-step temporal sequence, frozen QRC, and multinomial logistic readout. The script also reports raw-pixel logistic regression and a matched-dimensional fixed random-feature baseline, so expressivity is not inferred from feature dimension alone. The configuration is intentionally modest so judges can reproduce it.

## 10. IBM QPU execution

Read `docs/IBM_QPU_GUIDE.md` first. Never place an API key in the repository.

Prepare circuits without submitting:

```bash
python scripts/run_phase3_ibm.py \
  --qrc-config configs/gate_qrc_5q_hardware.yaml \
  --windows 10 --shots 1024 --prepare-only
```

After IBM authentication, run a 5-window smoke test:

```bash
python scripts/run_phase3_ibm.py \
  --qrc-config configs/gate_qrc_5q_hardware.yaml \
  --windows 5 --shots 512 --backend auto
```

Then the final targeted validation:

```bash
python scripts/run_phase3_ibm.py \
  --qrc-config configs/gate_qrc_5q_hardware.yaml \
  --windows 40 --shots 1024 --backend auto \
  --optimization-level 3 --resilience-level 1
```

The script records backend, job IDs, logical and transpiled depth, gate counts, shots, Runtime metadata, and prediction metrics.

## 11. Optional second QPU through qBraid credits

List devices visible to the team account:

```bash
python scripts/list_qbraid_devices.py
```

Choose one **gate-model** device with at least five qubits and a compatible native gate set. Then:

```bash
python scripts/run_phase3_qbraid_qpu.py \
  --device-qrn '<QRN_FROM_DEVICE_LIST>' \
  --windows 20 --shots 1024 --batch-size 10
```

Recommended priority:

1. IBM QPU - primary, because the package uses EstimatorV2 for direct Z/ZZ expectation values.
2. One qBraid-managed trapped-ion or superconducting device - secondary cross-platform validation if credits permit.
3. Neutral-atom analog hardware - stretch only; it requires a separate analog Hamiltonian implementation and should not displace the required gate-QRC/noise/MNIST work.

D-Wave and QCi are not primary choices for this project because the implemented reservoir is a driven gate-model dynamical feature map, not a QUBO/annealing optimization problem.

## 12. Aggregate, compare, and generate figures

```bash
python scripts/aggregate_phase3.py
python scripts/compare_predictions.py \
  results/predictions/<QRC_PREDICTIONS>.csv \
  results/predictions/<TCN_PREDICTIONS>.csv --loss qlike
python scripts/make_phase3_figures.py
```

## 13. Build the five-page paper

```bash
cd writeup
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

`scripts/aggregate_phase3.py` regenerates `writeup/generated_results.tex` and the compact result table automatically. Edit only the prose interpretation after checking the generated numbers. Keep the official cover page as the first page of the final PDF and do not recreate it.

Export the **filled official cover template** to PDF, then prepend it without altering the page:

```bash
python scripts/assemble_final_pdf.py \
  --cover docs/GIC_2026_Cover_Filled.pdf \
  --body writeup/main.pdf \
  --output FraQTech_Phase3_Writeup.pdf
```

## 14. Validate and package

```bash
python scripts/validate_submission.py --require-results
python scripts/package_submission.py \
  --team FraQTech --challenge DynamicSystemsForecasting --require-results
```

Expected final filename:

```text
FraQTech_DynamicSystemsForecasting_Phase3.zip
```

## Reproducibility rules

- Never tune on test data.
- Never overwrite a result without preserving its configuration and seed.
- Report cached and uncached feature-generation costs separately.
- Distinguish exact simulation, finite-shot simulation, noisy simulation, and physical QPU results.
- Do not claim universal quantum advantage. Report the regime where the QRC is competitive and where it is not.
- Preserve QPU job IDs and backend calibration metadata.

## Known limitations

- Exact statevector cost grows exponentially with qubit count.
- Hardware evaluation uses a targeted subset because each temporal checkpoint is a circuit.
- The continuous-variable pilot model is a simulator reference, not a qubit/QPU result.
- The new gate-QRC is SOTA-inspired and validation-selected; it must earn any performance claim through the supplied experiments.
- Public market data can be revised. The final repository should include the frozen processed CSV and checksum.
