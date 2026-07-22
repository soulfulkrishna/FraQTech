# Platform strategy

## Primary: IBM Quantum through Qiskit Runtime

Why it is primary:

- The gate-QRC is natively expressed as Qiskit circuits.
- EstimatorV2 directly returns expectation values of the Z and ZZ observables used as reservoir features.
- Transpilation metadata gives defensible depth and two-qubit gate counts.
- Dynamical decoupling and resilience settings can be recorded and reproduced.
- IBM Open Plan provides an independent access route even if challenge credits are delayed.

Use IBM for one carefully bounded, end-to-end hardware result. Do not spend the budget on a full test set.

## Secondary: one qBraid-managed gate QPU

Use `scripts/list_qbraid_devices.py` after the team allocation appears. Choose a device by:

1. availability to the team account;
2. at least five usable qubits;
3. native entangling gate and topology compatible with the selected QRC;
4. transpiled depth and estimated credit cost;
5. ability to return shot counts for Z/ZZ post-processing.

A trapped-ion device is attractive when all-to-all connectivity removes SWAP overhead. A superconducting device is attractive as a cross-vendor robustness test. The paper should report whichever is actually available, not promise a named device before the account exposes it.

## Stretch: neutral-atom analog QRC

Neutral-atom hardware is highly relevant to large-scale QRC literature, but a faithful analog Rydberg implementation is a second architecture. Add it only after the required gate-QRC, noise, MNIST, and reproducibility work is complete.

## Platforms not recommended for the main result

- D-Wave: designed around annealing/hybrid optimization, whereas this project needs driven temporal quantum dynamics and observable trajectories.
- QCi Dirac: similarly not the natural execution target for this gate-reservoir feature map.

The sentence in the supplied challenge PDF that says all Phase 3 work must run on QCi Dirac-3 conflicts with the rest of the QRC challenge. Ask the organizers to confirm that it is a template carry-over before submission.
