# Final submission checklist

## Scientific

- [ ] Freeze VIX dataset CSV and checksum.
- [ ] Freeze gate-QRC architecture using validation only.
- [ ] Complete five classical seeds.
- [ ] Complete three QRC seeds for 5/10/15 qubits or document bounded 15-qubit limits.
- [ ] Run matched random-feature and QRC ablations.
- [ ] Run 256/1024/4096-shot comparison.
- [ ] Run depolarizing and amplitude-damping study.
- [ ] Run at least one physical QPU result and preserve job IDs.
- [ ] Complete MNIST 5/10/15-qubit benchmark.
- [ ] Generate prediction-level Diebold-Mariano comparisons.

## Write-up

- [ ] Official cover page is first and unmodified.
- [ ] Main body is at most five pages, 11-point Times New Roman, single-spaced.
- [ ] References are outside the five-page count.
- [ ] All headline numbers exist in `results/raw` and `results/predictions`.
- [ ] Qubit count, depth, two-qubit gates, shots, runtime, and backend appear explicitly.
- [ ] Distinguish exact simulation, noisy simulation, and QPU results.
- [ ] State that the continuous-variable pilot is simulation-only.
- [ ] Avoid universal quantum-advantage claims.
- [ ] Disclose generative-AI assistance.

## Repository

- [ ] Replace `YOUR_GITHUB_REPO.git` in README Launch button.
- [ ] Run `python scripts/smoke_test.py` in a fresh qBraid environment.
- [ ] Run `pytest -q`.
- [ ] Run `python scripts/validate_submission.py --require-results`.
- [ ] No API keys, tokens, CRNs, or personal paths are committed.
- [ ] README commands work without editing source files.
- [ ] qBraid Skill is installed/tested using the current CLI available in the challenge workspace.
- [ ] Include known limitations.

## Packaging

- [ ] Final PDF has official cover page + <=5 body pages + references.
- [ ] Final source-code folder includes the frozen processed dataset or deterministic builder.
- [ ] Final ZIP is named `FraQTech_DynamicSystemsForecasting_Phase3.zip`.
- [ ] Extract the ZIP into a clean directory and rerun the smoke test.
- [ ] Upload before July 26, 2026, 11:59 PM EST.
