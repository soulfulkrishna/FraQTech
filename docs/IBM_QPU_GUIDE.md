# IBM Quantum hardware guide

## A. Install in a trusted qBraid Lab environment

```bash
pip install 'qiskit[all]~=2.4.0' 'qiskit-ibm-runtime~=0.46.1'
```

## B. Create credentials

1. Log in to IBM Quantum Platform.
2. Create/copy the API key.
3. Copy the instance CRN from the Instances page.
4. In the qBraid terminal, save credentials once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    token="<IBM_API_KEY>",
    instance="<INSTANCE_CRN>",
    overwrite=True,
)
```

Do not paste either secret into a notebook committed to Git.

Verify:

```bash
python - <<'PY'
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(service.active_instance())
for backend in service.backends(simulator=False, operational=True):
    print(backend.name, backend.num_qubits)
PY
```

## C. Prepare before spending QPU time

```bash
python scripts/run_phase3_ibm.py \
  --qrc-config configs/gate_qrc_5q_hardware.yaml \
  --windows 10 --shots 1024 --prepare-only
```

Inspect `results/hardware/ibm_prepare_seed0.json`. Confirm:

- number of circuits;
- total shot estimate;
- logical depth;
- no accidental 15-qubit or full-test run.

## D. Run in three stages

### 1. Credential/backend smoke test

```bash
python scripts/run_phase3_ibm.py \
  --windows 2 --shots 256 --backend auto
```

### 2. Scientific smoke test

```bash
python scripts/run_phase3_ibm.py \
  --windows 5 --shots 512 --backend auto
```

### 3. Final run

```bash
python scripts/run_phase3_ibm.py \
  --windows 40 --shots 1024 --backend auto \
  --optimization-level 3 --resilience-level 1
```

The script selects the least-busy operational backend with enough qubits unless `--backend` is specified. It maps observables to the transpiled circuit layout, uses EstimatorV2, enables XY4 dynamical decoupling by default, and stores job IDs and resource metadata.

## E. Retrieve interrupted jobs

Qiskit Runtime jobs persist after the notebook or laptop closes. Copy every job ID from `results/hardware/ibm_metadata_seed*.json`. To retrieve manually:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
job = service.job("<JOB_ID>")
print(job.status())
result = job.result()
```

## F. What to report

- backend name and execution date;
- number of QRC qubits;
- number of windows/checkpoint circuits;
- shots per circuit and total shots;
- optimization and resilience level;
- dynamical-decoupling sequence;
- logical and transpiled depth;
- mean two-qubit gate count;
- wall-clock and available job metrics;
- raw hardware NRMSE/QLIKE and simulator-to-hardware degradation;
- limitations due to subset size and queue/credit constraints.

## G. Budget control

The checkpoint construction deliberately uses six temporal bins, one virtual node, and a ring topology on hardware. Begin with 2-5 windows. Only scale after observing the transpiled depth and actual usage. A clean 20-40-window validation is more useful than an incomplete 100-window job.
