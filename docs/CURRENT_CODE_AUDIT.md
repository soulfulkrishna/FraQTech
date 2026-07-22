# Audit of the uploaded benchmark code

## What was already solid

The uploaded project already had a useful experiment spine:

- chronological datasets and repeatable seed-based runs;
- persistence, ARIMA, ridge-lag, ESN, MLP/RNN/GRU/LSTM/TCN/Transformer baselines;
- CV-QRC and a legacy gate-QRC encoder with multiple classical readouts;
- cache-aware QRC feature extraction;
- energy/carbon/resource logging;
- JSON raw logs, resume scripts, aggregation, and metric auditing.

The original files are retained under `legacy/` without modification.

## Why the old scripts were not sufficient for Phase 3

1. **Explicit target handling.** The CSV loader read the `value` series into both `x` and `y`, then recreated the target via generic windowing. The Phase 3 loader now consumes the explicit future `target` column.
2. **Validation separation.** The old runner joined train and validation before normalization/training. Phase 3 fits scaling and hyperparameters using training/validation only, freezes the configuration, then evaluates the test split once.
3. **Finance-specific evaluation.** The old metrics were RMSE, NRMSE, and MAE. Phase 3 adds QLIKE, Mincer-Zarnowitz calibration, Diebold-Mariano comparison, regime classification, and transition recall.
4. **Required baselines.** GARCH and HAR-RV were absent. They are now included alongside ESN and TCN.
5. **Hardware execution.** The old gate-QRC was a local exact statevector model; its shot field was metadata rather than a physical execution path. Phase 3 includes matching Qiskit circuits, Aer noise, IBM Runtime, and qBraid-managed QPU adapters.
6. **Qubit/noise/MNIST matrices.** The old experiment plan did not implement the challenge's 5/10/15-qubit, finite-shot, realistic-noise, and MNIST requirements.
7. **Prediction-level artifacts.** Phase 3 saves every forecast/probability with timestamps so headline comparisons can be independently recalculated.
8. **Portability.** The new package uses project-relative paths and contains a qBraid-first README, smoke test, requirements, manifest, and packaging validator.

## Important scientific correction

A preliminary GARCH implementation initially mapped daily GARCH volatility to five-day rolling volatility using the test median. This would be test leakage. The final package estimates that scalar mapping on the validation split only.
