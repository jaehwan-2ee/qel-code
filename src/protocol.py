"""The QEL training protocol: its constants, the scale S, and the start of a fit.

A fit has three stages, each epoch being one update on the full TRAIN batch:
  1. encoder warm-up   L-BFGS fit of the encoder to the true coefficients (src/lbfgs.py);
  2. angle warm-up     Adam on the circuit angles with the encoder fixed (src/stages.py);
  3. joint training    Adam on the encoder and the circuit angles (src/stages.py, gradient in src/spsa.py); the
                       reported checkpoint is the best epoch on the VALID energy under the early-stopping rule.
"""

import math
from typing import NamedTuple

import numpy as np
import torch

from src import ba_circuit, bm_circuit, kp_circuit
from src.encoder import n_encoder

CIRCUIT = {"ba": ba_circuit, "kp": kp_circuit, "bm": bm_circuit}

LBFGS_ITERATIONS = 50
ANGLE_WARMUP_EPOCHS = 50
JOINT_EPOCHS = 100
SHOTS = 2048  # per TEST instance

# SPSA perturbation of update k (from 0): c / (k + 1)^SPSA_EXPONENT
SPSA_C_ANGLES = 0.025
SPSA_C_COEFFICIENTS = 0.025  # in units of S
SPSA_EXPONENT = 0.101

# Initial angles: gamma uniform on [0, pi / denominator), beta on (-pi / denominator, 0].
INITIAL_ANGLE_DENOMINATOR = {"ba": 4.0, "kp": 0.5, "bm": 8.0}
GLOROT_STREAM = 117300  # labels the NumPy stream of the initial encoder weights


class Spec(NamedTuple):
    """What a circuit needs besides the parameters and the data of an instance."""

    problem: str
    hidden: int
    output_activation: str
    S: float
    weights: np.ndarray | None  # KP item weights


def true_coefficients(problem, y):
    """Objective coefficients in the true values y: the summed user values of each website (BA), y itself (KP, BM)."""
    return y.sum(-1) if problem == "ba" else y


def scale(problem, y_train):
    """S = max(1, max |c| / 2) over the true coefficients c of the TRAIN instances. Every energy is divided by S."""
    return max(1.0, float(np.abs(true_coefficients(problem, y_train)).max()) / 2)


def initial_angles(feature_dim, layers, seed, denominator):
    """Angles (gamma, beta) of every layer, float32: uniform on [0, pi / denominator), the mixer angles beta negated.

    A CUDA generator seeded by `seed` draws 1 + feature_dim + 2 * layers numbers in one call and the angles are the
    last 2 * layers of them. The leading numbers are unused, but the size of the call fixes the values."""
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(seed)
    draw = torch.rand(1 + feature_dim + 2 * layers, device="cuda:0", generator=generator)
    angles = (draw[1 + feature_dim :] * torch.pi * (1.0 / denominator)).reshape(layers, 2)
    angles[:, -1] *= -1
    return angles.reshape(-1).cpu().numpy()


def initial_parameters(problem, feature_dim, hidden, layers, seed, output_scale):
    """The start [encoder | circuit angles], float32. It depends on the architecture and the model seed only.
    The encoder is Glorot-uniform with zero biases; its output layer is multiplied by output_scale."""
    encoder = np.zeros(n_encoder(feature_dim, hidden), np.float32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, GLOROT_STREAM, hidden, feature_dim]))
    encoder[: hidden * feature_dim] = rng.uniform(
        -math.sqrt(6 / (feature_dim + hidden)), math.sqrt(6 / (feature_dim + hidden)), hidden * feature_dim
    )
    encoder[hidden * feature_dim + hidden : -1] = rng.uniform(
        -math.sqrt(6 / (hidden + 1)), math.sqrt(6 / (hidden + 1)), hidden
    )
    encoder[hidden * (feature_dim + 1) :] *= np.float32(output_scale)
    return np.concatenate((encoder, initial_angles(feature_dim, layers, seed, INITIAL_ANGLE_DENOMINATOR[problem])))
