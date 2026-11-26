"""BM circuit, simulated exactly on the n! assignments of an n x n bipartite matching (n = 10).

The feasible decisions are the perfect matchings, so the state lives on the assignments in lexicographic order and
starts as their uniform superposition. Each layer (gamma, beta) applies the phase of the predicted coefficients and
then, for every pair of rows in a round-robin order, the rotation cos(beta) I - i sin(beta) SwapRows. On the
assignments this equals the product of the four-qubit exchanges over all pairs of columns.
"""

import math
from collections import Counter
from functools import cache, lru_cache
from itertools import chain
from itertools import permutations as _permutations

import numpy as np
import torch

from data import BM_NODES
from src.encoder import predict

DEVICE = "cuda:0"  # the first visible GPU; 10! amplitudes per state


@cache
def permutations(n):
    """All n! assignments as an (n!, n) array in lexicographic order: entry [k, r] is the column of row r."""
    out = np.fromiter(chain.from_iterable(_permutations(range(n))), np.uint8, count=math.factorial(n) * n).reshape(
        -1, n
    )
    return out


def matching_rounds(n):
    """Circle-method edge coloring of K_n (even n)."""
    vertices = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append(tuple((min(a, b), max(a, b)) for a, b in zip(vertices[: n // 2], reversed(vertices[n // 2 :]))))
        vertices = [vertices[0], vertices[-1], *vertices[1:-1]]
    return tuple(rounds)


@lru_cache(maxsize=4)
def _tables(n, device):
    """The assignments, and for every pair of rows the index of each assignment after swapping the two rows
    (int32: 653 MB for the 45 pairs of n = 10)."""
    p = permutations(n)
    size = len(p)
    digits = np.zeros_like(p)
    for r in range(n):
        for k in range(r + 1, n):
            digits[:, r] += p[:, k] < p[:, r]
    maps = np.empty((n * (n - 1) // 2, size), np.int32)
    for j, (r, s) in enumerate(pair for group in matching_rounds(n) for pair in group):
        a, b = p[:, r], p[:, s]
        dr = (a < b).astype(np.int64) - digits[:, r]
        ds = -digits[:, s].astype(np.int64)
        for k in range(r + 1, n):
            if k != s:
                dr += p[:, k] < b
        for k in range(s + 1, n):
            ds += p[:, k] < a
        rank = np.arange(size, dtype=np.int64)
        rank += dr * math.factorial(n - 1 - r) + ds * math.factorial(n - 1 - s)
        for k in range(r + 1, s):
            rank += ((a < p[:, k]).astype(np.int64) - (b < p[:, k])) * math.factorial(n - 1 - k)
        maps[j] = rank
    return (torch.tensor(p.astype(np.int64), device=device), torch.as_tensor(maps, device=device))


def assignment_values(values):
    """Sum of the chosen entries of an (n, n) matrix, for every assignment."""
    n = values.shape[0]
    p, _ = _tables(n, str(values.device))
    out = torch.zeros(len(p), dtype=values.dtype, device=values.device)
    for r in range(n):
        out = out + values[r].index_select(0, p[:, r])
    return out


# cos(beta) state - i sin(beta) state[index]. Compiled for speed; needs a C compiler on PATH.
@torch.compile(fullgraph=True)
def _rotate(state, index, cosine, sine):
    # Real views let Inductor fuse complex gather/arithmetic into one kernel.
    value = torch.view_as_real(state)
    swapped = value.index_select(0, index)
    return torch.view_as_complex(
        torch.stack((cosine * value[:, 0] + sine * swapped[:, 1], cosine * value[:, 1] - sine * swapped[:, 0]), -1)
    )


@torch.no_grad()
def _evolve(angles, phase, swaps):
    """From the uniform superposition of all assignments: per layer the phase, then one rotation per pair of rows."""
    state = torch.zeros_like(phase, dtype=torch.complex128)
    state.fill_(1 / math.sqrt(phase.numel()))
    for gamma, beta in angles.reshape(-1, 2):
        state = state * torch.exp(1j * gamma * phase)
        cosine, sine = torch.cos(beta), torch.sin(beta)
        for index in swaps:
            state = _rotate(state, index, cosine, sine)
    return state


def probabilities(angles, coefficients, scale):
    """Probability of every assignment; `coefficients` is the (n, n) matrix of one instance."""
    n = coefficients.shape[0]
    _, swaps = _tables(n, str(coefficients.device))
    # the phase of an assignment, including the global phase of the per-qubit RZ gates
    phase = (assignment_values(coefficients) - coefficients.sum() / 2) / scale
    return _evolve(angles, phase, swaps).abs().square()


def target(y):
    """Objective of every assignment."""
    return assignment_values(y.reshape(BM_NODES, BM_NODES))


def instance_energy(angles, coefficients, objective, spec):
    """Expected energy of one instance: minus the expected objective, divided by S."""
    prob = probabilities(angles, coefficients.reshape(BM_NODES, BM_NODES), spec.S)
    return -(prob * objective).sum() / spec.S


def _draw(prob, n, shots, seed):
    """Shot counts of one instance, keyed by the n x n assignment matrix written row by row: inverse-CDF sampling with a
    NumPy PCG64 stream seeded by `seed`."""
    p = np.asarray(prob, np.float64)
    cdf = np.cumsum(p / p.sum())
    cdf[-1] = 1
    indices = np.searchsorted(cdf, np.random.default_rng(seed).random(shots), side="right")
    counts = Counter()
    for index, count in zip(*np.unique(indices, return_counts=True)):
        bits = np.zeros(n * n, dtype=np.uint8)
        bits[np.arange(n) * n + permutations(n)[index]] = 1
        counts["".join(bits.astype(str))] = int(count)
    return dict(counts)


def sample_counts(params, X, spec, shots, seed):
    """Shot counts of every TEST instance, and the predicted coefficients they were drawn with.
    Instance r uses the stream seed + r."""
    angles, predicted = predict(params, X, spec, DEVICE)
    counts = [
        _draw(probabilities(angles, row.reshape(BM_NODES, BM_NODES), spec.S).cpu().numpy(), BM_NODES, shots, seed + r)
        for r, row in enumerate(predicted)
    ]
    return counts, predicted.cpu().numpy()
