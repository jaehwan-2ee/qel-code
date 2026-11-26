"""BA circuit: five qubits, one per website, simulated exactly as a statevector.

The feasible decisions are the one-hot bitstrings, so the circuit starts in the W state and uses the mixer
exp(-i beta |W><W|), which keeps the state in their span. The state is built from X, RY and CNOT gates.
"""

import math
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.encoder import predict

N_WEBSITES = 5
N_STATES = 2**N_WEBSITES
ONE_HOT = [1 << wire for wire in range(N_WEBSITES)]  # basis-state index of "website `wire` chosen"
# A stored mixer angle beta stands for exp(-i beta |W><W|). Between the inverse W tree and the W tree the gate program
# applies the phase exp(i t) to |1> and exp(-i t) to |0> of wire 0; after conjugation by the W tree this equals
# exp(2 i t |W><W|) on the one-hot subspace, up to a global phase, hence t = -beta / 2.
MIXER_ANGLE_SCALE = -0.5
DEVICE = "cpu"


class Gate(NamedTuple):
    name: str
    wires: tuple[int, ...]
    angle: float | None = None  # rotation angle of an RY gate


def gate(name, *wires, angle=None):
    return Gate(name, tuple(wires), None if angle is None else float(angle))


def inverse(operations):
    return tuple(op._replace(angle=None if op.angle is None else -op.angle) for op in reversed(operations))


def givens(a, b, alpha):
    """Move sin(alpha) of |10> to |01>: XXPlusYY(2alpha,+pi/2), two CX."""
    return (
        gate("RY", b, angle=math.pi / 2),
        gate("CNOT", b, a),
        gate("RY", a, angle=alpha),
        gate("RY", b, angle=alpha),
        gate("CNOT", b, a),
        gate("RY", b, angle=-math.pi / 2),
    )


@lru_cache(maxsize=1)
def w_tree():
    return (
        *givens(0, 4, math.acos(math.sqrt(4 / 5))),
        *givens(0, 2, math.pi / 4),
        *givens(0, 1, math.pi / 4),
        *givens(2, 3, math.pi / 4),
    )


@lru_cache(maxsize=1)
def w_preparation():
    """Preparation of the W state from |00000>: X on wire 0, then the W tree, whose first split needs one CNOT only
    (the root is known to be 1 and leaf 4 to be 0)."""
    prep = w_tree()
    return (gate("X", 0), gate("RY", 4, angle=2 * math.acos(math.sqrt(4 / 5))), gate("CNOT", 4, 0), *prep[6:])


@lru_cache(maxsize=1)
def _indices():
    """Per wire: the index pairs an RY acts on, and +1 where the qubit is 1, -1 where it is 0; shared by every call."""
    x = np.arange(N_STATES)
    pairs = []
    for wire in range(N_WEBSITES):
        lo = x[(x >> wire & 1) == 0]
        hi = lo | (1 << wire)
        pairs.append((torch.tensor(lo), torch.tensor(hi), torch.tensor(np.argsort(np.stack([lo, hi], 1).reshape(-1)))))
    return x, pairs, torch.tensor(((x[None, :] >> np.arange(N_WEBSITES)[:, None]) & 1) * 2 - 1, dtype=torch.float64)


def probabilities(angles, coefficients, scale):
    """Probabilities of the 32 basis states for a batch of instances: the W state, then for every layer (gamma, beta)
    the phase separator exp(i gamma c_j x_j / S) on the bit x_j of every website j (up to a global phase) and the
    mixer exp(-i beta |W><W|)."""
    x, pairs, signs = _indices()
    initial, prep = w_preparation(), w_tree()
    state = torch.zeros((len(coefficients), N_STATES), dtype=torch.complex128)
    state[:, 0] = 1

    def fixed(ops, current):
        for op in ops:
            w = op.wires
            if op.name == "X":
                current = current[:, torch.tensor(x ^ (1 << w[0]))]
            elif op.name == "CNOT":
                permutation = x ^ (((x >> w[0]) & 1) << w[1])
                current = current[:, torch.tensor(permutation)]
            else:  # RY
                lo, hi, order = pairs[w[0]]
                a, b = current[:, lo], current[:, hi]
                c, s = math.cos(op.angle / 2), math.sin(op.angle / 2)
                current = torch.stack((c * a - s * b, s * a + c * b), -1).reshape(len(coefficients), N_STATES)[:, order]
        return current

    state = fixed(initial, state)
    for gamma, beta in angles.reshape(-1, 2):
        for wire in range(N_WEBSITES):
            angle = gamma * coefficients[:, wire] / scale
            state = state * torch.exp(0.5j * angle[:, None] * signs[wire])
        # the mixer: undo the W tree, a phase on wire 0, redo the tree
        state = fixed(inverse(prep), state)
        state = state * torch.exp(1j * (MIXER_ANGLE_SCALE * beta) * signs[0])[None, :]
        state = fixed(prep, state)
    return state.abs().square()


def target(y):
    """Objective of choosing each website: the sum of its user values."""
    return y.sum(-1)


def instance_energy(angles, coefficients, website_values, spec):
    """Expected energy of one instance: minus the expected objective, divided by S."""
    return -(probabilities(angles, coefficients[None], spec.S)[:, ONE_HOT] * website_values[None]).sum() / spec.S


def _draw(probs, uniforms):
    """Shot counts of one instance. The five qubits are sampled one after another, each from its conditional
    distribution given the bits already drawn, with one uniform number per shot and qubit."""
    uniforms = np.asarray(uniforms)
    bits = (np.arange(N_STATES)[:, None] >> np.arange(N_WEBSITES - 1, -1, -1)) & 1
    match = np.ones((len(uniforms), N_STATES), bool)
    chosen = np.zeros((len(uniforms), N_WEBSITES), np.int8)
    for wire in range(N_WEBSITES):
        mass = np.where(match, probs, 0.0)
        bit = uniforms[:, wire] > (mass * (1 - bits[:, wire])).sum(1) / mass.sum(1)  # exceeds P(bit = 0)
        chosen[:, wire] = bit
        match &= bits[:, wire][None, :] == bit[:, None]
    counts = {}
    for row in chosen:
        key = "".join(str(int(bit)) for bit in row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def sample_counts(params, X, spec, shots, seed):
    """Shot counts of every TEST instance, and the predicted coefficients they were drawn with.
    One JAX key seeded by `seed` is split once per instance."""
    angles, predicted = predict(params, X, spec, DEVICE)
    prob = probabilities(angles, predicted, spec.S).numpy()
    reverse = np.array([int(f"{i:05b}"[::-1], 2) for i in range(N_STATES)])  # wire 0 becomes the first bit of a shot
    key = jax.random.PRNGKey(seed)
    counts = []
    for row in prob:
        key, call = jax.random.split(key)
        uniforms = jax.random.uniform(call, (shots, N_WEBSITES), dtype=jnp.float32)
        counts.append(_draw(row[reverse], uniforms))
    return counts, predicted.numpy()
