---
name: fraqtech-vix-qrc
version: 1.0.0
description: Reproduce FraQTech Phase 3 VIX forecasting, gate-QRC scaling, noise, MNIST, and targeted QPU validation.
---

# FraQTech VIX QRC Skill

## Purpose

Navigate and execute the Phase 3 workflow without modifying source code. Prefer the smallest command that answers the request. Never submit to a paid/QPU backend without explicit human confirmation.

## Safety rules

- Never print, save, or commit API tokens or IBM instance CRNs.
- Run `python scripts/smoke_test.py` before any long experiment.
- Run IBM/qBraid scripts with `--prepare-only` or a 2-5-window smoke test first.
- Do not use the test split for hyperparameter or architecture selection.
- Do not claim a result unless the corresponding JSON and prediction CSV exist.

## Commands

### Inspect project

```bash
python scripts/validate_submission.py
```

### Build data

```bash
python scripts/build_phase3_finance_data.py
```

### Smoke test

```bash
python scripts/reproduce_all.py --profile smoke --skip-data
```

### Classical benchmark

```bash
python scripts/run_phase3_classical.py --seed 0 --profile full
```

### Gate-QRC architecture selection

```bash
python scripts/select_gate_qrc_architecture.py --profile smoke --seed 0
```

### Gate-QRC simulator

```bash
python scripts/run_phase3_qrc_sim.py --qrc-config configs/gate_qrc_5q.yaml --readout ridge --seed 0
```

### Scaling

```bash
python scripts/run_phase3_scaling.py --qubits 5,10,15 --seeds 0,1,2 --readouts ridge,tcn
```

### Noise

```bash
python scripts/run_phase3_noise.py --seed 0 --sample-windows 100 --shots 1024
```

### MNIST

```bash
python scripts/run_phase3_mnist.py --qubits 5 --seed 0
```

### IBM preparation only

```bash
python scripts/run_phase3_ibm.py --windows 5 --shots 512 --prepare-only
```

### Aggregate

```bash
python scripts/aggregate_phase3.py
python scripts/make_phase3_figures.py
```

## Expected outputs

- Metrics: `results/raw/*.json`
- Predictions: `results/predictions/*.csv`
- Hardware metadata and job IDs: `results/hardware/*`
- Aggregated tables: `results/summaries/*`
- Figures: `results/figures/*`

## Failure handling

- Missing dataset: run the data builder.
- Out of memory: reduce `--max-train`, batch size, or qubit count; do not silently change the reported configuration.
- QPU credentials absent: stop after `--prepare-only` and explain the authentication step.
- Device unavailable: list qBraid/IBM devices and choose a compatible operational backend only with human approval.
