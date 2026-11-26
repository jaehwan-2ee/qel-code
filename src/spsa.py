"""SPSA gradient of the mean TRAIN energy. Both blocks of the model are differentiated by two-sided simultaneous
perturbation, which needs energies only:

  circuit angles  one Rademacher direction per update, shared by all TRAIN instances (2n energies);
  encoder         for every instance a Rademacher direction on its predicted coefficients (2n energies); this estimates
                  the derivative with respect to the coefficients, which the MLP then propagates analytically.
"""

import math

import numpy as np
import torch

from src.encoder import coefficients, jacobian_product, n_encoder
from src.protocol import SPSA_C_ANGLES, SPSA_C_COEFFICIENTS, SPSA_EXPONENT

# Labels of the NumPy streams of the directions. The angle warm-up and the joint stage use different angle streams.
COEFFICIENT_STREAM, JOINT_ANGLE_STREAM, WARMUP_ANGLE_STREAM = 73300, 73301, 73302


def _signs(size, *stream):
    """Rademacher direction from the NumPy stream seeded by the integers in `stream`."""
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    return (2 * rng.integers(0, 2, size=size, dtype=np.int8) - 1).astype(np.float32)


def _project(problem, direction):
    """Remove the part of a coefficient direction that only shifts the global phase: a common shift of the five BA
    coefficients, and row and column shifts of the BM matrix."""
    if problem == "ba":
        return direction - direction.mean()
    if problem == "bm":
        n = math.isqrt(len(direction))
        d = direction.reshape(n, n)
        return (d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()).reshape(-1)
    return direction


class InstanceEnergy:
    """Expected energy of TRAIN instance `row` for given circuit angles and coefficients."""

    def __init__(self, circuit, y, spec):
        self.circuit, self.spec = circuit, spec
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=circuit.DEVICE)
        self.row = None

    def __call__(self, angles, coefficients, row):
        if row != self.row:  # the four energies of an instance share its target
            self.row, self.target = row, self.circuit.target(self.y[row])
        return self.circuit.instance_energy(angles.double(), coefficients, self.target, self.spec)


class Gradient:
    """The SPSA gradients of update number `update`, which starts at 0."""

    def __init__(self, circuit, X, y, spec, seed):
        self.spec, self.seed, self.update = spec, seed, 0
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=circuit.DEVICE)
        self.n = n_encoder(X.shape[-1], spec.hidden)
        self.energy = InstanceEnergy(circuit, y, spec)

    def _angle_perturbation(self, p, stream):
        """(c, direction, angles + c direction, angles - c direction)."""
        c = SPSA_C_ANGLES / (self.update + 1) ** SPSA_EXPONENT
        direction = torch.as_tensor(_signs(len(p) - self.n, self.seed, self.update, 1, 0, stream), device=p.device)
        angles = p[self.n :]
        return c, direction.double(), angles + c * direction, angles - c * direction

    def _encoder(self, p):
        """The encoder parameters as a float64 tensor, and the routine that predicts the coefficients of one row."""
        encoder = p[: self.n].double()
        hidden, activation = self.spec.hidden, self.spec.output_activation
        return encoder, lambda row: coefficients(encoder, self.X[row], hidden, activation)

    def angles(self, p):
        """Gradient with respect to the circuit angles only (angle warm-up)."""
        c, direction, angles_plus, angles_minus = self._angle_perturbation(p, WARMUP_ANGLE_STREAM)
        _, predict = self._encoder(p)
        e_angles_plus, e_angles_minus = [], []
        for row in range(len(self.X)):
            predicted = predict(row)
            e_angles_plus.append(self.energy(angles_plus, predicted, row))
            e_angles_minus.append(self.energy(angles_minus, predicted, row))
        self.update += 1
        gradient = (torch.stack(e_angles_plus).mean() - torch.stack(e_angles_minus).mean()) / (2 * c) * direction
        return gradient.float()

    def joint(self, p):
        """(mean of the evaluated energies, float32 gradient) with respect to [encoder | circuit angles]."""
        n = self.n
        c, direction, angles_plus, angles_minus = self._angle_perturbation(p, JOINT_ANGLE_STREAM)
        encoder, predict = self._encoder(p)
        c_coefficients = SPSA_C_COEFFICIENTS * self.spec.S / (self.update + 1) ** SPSA_EXPONENT
        encoder_gradient = torch.zeros(n, dtype=torch.float64, device=p.device)
        e_coefficients_plus, e_coefficients_minus, e_angles_plus, e_angles_minus = [], [], [], []
        for row in range(len(self.X)):
            predicted = predict(row)
            signs = _signs(len(predicted), self.seed, self.update, 0, row, COEFFICIENT_STREAM)
            d = _project(self.spec.problem, torch.as_tensor(signs, device=p.device).double())
            e_coefficients_plus.append(self.energy(p[n:], predicted + c_coefficients * d, row))
            e_coefficients_minus.append(self.energy(p[n:], predicted - c_coefficients * d, row))
            slope = (e_coefficients_plus[-1] - e_coefficients_minus[-1]) / (2 * c_coefficients)
            encoder_gradient += jacobian_product(
                encoder, self.X[row], self.spec.hidden, self.spec.output_activation, slope * d
            )
            e_angles_plus.append(self.energy(angles_plus, predicted, row))
            e_angles_minus.append(self.energy(angles_minus, predicted, row))
        self.update += 1
        energies = [e_coefficients_plus, e_coefficients_minus, e_angles_plus, e_angles_minus]
        means = torch.stack([torch.stack(e).mean() for e in energies])
        gradient = torch.cat((encoder_gradient / len(self.X), (means[2] - means[3]) / (2 * c) * direction))
        return float(means.mean()), gradient.float()
