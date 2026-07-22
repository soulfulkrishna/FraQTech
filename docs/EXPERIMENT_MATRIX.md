# Frozen Phase 3 experiment matrix

## Main finance task

- Dataset: public VIX daily OHLC data, 2015-01-01 to 2025-12-31.
- Signal: five-day rolling standard deviation of log returns.
- Forecast: next trading day's five-day realized-volatility proxy.
- Lookback: 50 trading days.
- Split: chronological 60/20/20.
- Selection metric: validation QLIKE.
- Test: used once after architecture/readout selection.

## Main comparison

| Family | Models | Seeds |
|---|---|---:|
| Econometric/linear | Persistence, GARCH(1,1), HAR-RV, ridge-lag | 5 |
| Classical reservoir | ESN | 5 |
| Neural | TCN | 5 |
| QRC simulation | 5/10/15-qubit temporal Ising QRC + ridge/TCN | 3 |
| QRC ablations | Z-only, no re-upload, line topology, matched random features | 3 |

## Scaling

- Qubits: 5, 10, 15.
- Reservoir layers: selected on validation; expected 1-2.
- Temporal bins: selected from 4, 6, 8.
- Virtual nodes: 1 or 2.
- Observables: all single-Z and selected edge-ZZ correlations plus four global summaries.

## Noise/measurement

At 5 qubits, three seeds, 100 deterministic test windows:

| Condition | 1q depolarizing | 2q depolarizing | amplitude damping | shots |
|---|---:|---:|---:|---:|
| ideal-shot | 0 | 0 | 0 | 1024 |
| low | 0.001 | 0.005 | 0.001 | 1024 |
| medium | 0.005 | 0.010 | 0.005 | 1024 |
| high | 0.010 | 0.020 | 0.010 | 1024 |

Shot ablation additionally runs 256, 1024, and 4096 shots under the ideal-shot condition, isolating measurement variance. The noise sweep is held at 1024 shots.

## Hardware

- Primary: 5-qubit IBM QPU, 5-window smoke test then 40-window final subset.
- Stretch: 10 qubits only if transpiled depth, credits, and queue are acceptable.
- Secondary: one qBraid-managed gate QPU, 20-window subset.
- Record: backend, date, job ID, shots, logical/transpiled depth, two-qubit gate count, wall-clock, QPU metadata.

## MNIST

- Fixed 2,000 training and 500 test examples.
- Images resized to 8x8 and interpreted as eight temporal rows.
- 5/10/15 qubits.
- Logistic readout.
- Matched controls: raw 8x8 logistic regression and a fixed random nonlinear map with the same feature dimension as QRC.
- Accuracy, macro-F1, runtime, depth, feature dimension, and shot/noise degradation.

## Required headline tables/plots

1. VIX main performance table.
2. 5/10/15-qubit accuracy-resource table.
3. Noise/shot robustness table.
4. IBM and optional second-QPU table.
5. MNIST table.
6. Forecast trace with high-volatility regimes.
7. NRMSE versus qubits.
8. QLIKE versus total shots/noise.
9. Accuracy-resource Pareto plot.
