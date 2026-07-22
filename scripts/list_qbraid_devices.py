from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpu-only", action="store_true")
    parser.add_argument("--online-only", action="store_true")
    args = parser.parse_args()
    try:
        from qbraid import QbraidProvider
    except ImportError:
        try:
            from qbraid.runtime import QbraidProvider
        except ImportError as exc:
            raise RuntimeError("Install qbraid>=0.12 first") from exc
    provider = QbraidProvider()
    rows = []
    for device in provider.get_devices():
        raw_meta = getattr(device, "metadata", {})
        meta = raw_meta() if callable(raw_meta) else raw_meta
        if args.qpu_only and str(meta.get("device_type", "")).upper() != "QPU":
            continue
        status = str(meta.get("status", ""))
        if args.online_only and status.upper() != "ONLINE":
            continue
        rows.append(
            {
                "id": (lambda x: x() if callable(x) else x)(getattr(device, "id", meta.get("device_id"))),
                "type": meta.get("device_type"),
                "qubits": meta.get("num_qubits"),
                "status": status,
                "queue": meta.get("queue_depth"),
                "provider": meta.get("provider"),
            }
        )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
