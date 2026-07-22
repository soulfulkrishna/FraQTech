from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn


class PennyLaneQuantumLayer(nn.Module):
    """
    Small differentiable quantum feature layer.

    Input:
        x: [B, n_qubits]

    Output:
        expvals: [B, n_qubits]

    This is simulation-only via PennyLane default.qubit.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        circuit_type: Literal["basic", "ising"] = "basic",
        shots: int = 0,
    ) -> None:
        super().__init__()

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.circuit_type = circuit_type
        self.shots = int(shots)

        wires = list(range(self.n_qubits))

        # For differentiable training, use analytic statevector mode.
        # shots=0 means exact expectation values.
        dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=None if self.shots == 0 else self.shots,
        )

        if self.circuit_type == "ising":
            weight_shapes = {
                "weights": (self.n_layers, self.n_qubits, 4)
            }

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")

                for layer in range(self.n_layers):
                    for q in wires:
                        qml.RX(weights[layer, q, 0], wires=q)
                        qml.RY(weights[layer, q, 1], wires=q)
                        qml.RZ(weights[layer, q, 2], wires=q)

                    for q in wires:
                        qml.IsingXX(
                            weights[layer, q, 3],
                            wires=[q, (q + 1) % self.n_qubits],
                        )

                return [qml.expval(qml.PauliZ(q)) for q in wires]

        else:
            weight_shapes = {
                "weights": (self.n_layers, self.n_qubits, 3)
            }

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")

                for layer in range(self.n_layers):
                    for q in wires:
                        qml.RX(weights[layer, q, 0], wires=q)
                        qml.RY(weights[layer, q, 1], wires=q)
                        qml.RZ(weights[layer, q, 2], wires=q)

                    for q in wires:
                        qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

                return [qml.expval(qml.PauliZ(q)) for q in wires]

        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [B, n_qubits], got {tuple(x.shape)}")

        return self.layer(x)

    def circuit_depth_proxy(self) -> int:
        if self.circuit_type == "ising":
            return int(self.n_layers * (self.n_qubits * 3 + self.n_qubits))
        return int(self.n_layers * (self.n_qubits * 3 + self.n_qubits))