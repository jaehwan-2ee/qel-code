"""KP circuit, simulated exactly by dynamic programming over (item, accumulated weight).

State preparation: the items are scanned in the order of their predicted density (predicted coefficient / weight). An
item that still fits is taken with a prior probability, otherwise it is skipped, so every basis state with non-zero
amplitude is feasible. The prior is the warm start of Wilkening et al.: an item of the greedy solution along the scan
order is taken with probability (1 + n/4) / (2 + n/4), any other item with 1 / (2 + n/4).
Each layer (gamma, beta) applies the phase exp(i gamma phi(x)), phi(x) = sum_i s_i x_i / S with the predicted
coefficients s, and then the Grover mixer around the prepared state: a <- a - (1 - exp(-i beta)) E_p[a]. After L layers
the amplitude of x is sqrt(p(x)) a(x) with a(x) = sum_{j=0..L} alpha_j exp(i Gamma_j phi(x)), which the dynamic programs
below evaluate exactly.

The energies are evaluated in torch. The TEST shots are drawn in NumPy from the same terms (Gamma_j, alpha_j).
"""

import numpy as np
import torch

from data import KP_CAPACITY
from src.encoder import predict

DEVICE = "cpu"


def scan_order(s, w):
    """Items by descending predicted density s / w; ties keep the item order."""
    return np.argsort(-(np.asarray(s, np.float64) / np.asarray(w, np.float64)), kind="stable")


def take_probabilities(w, C):
    """Prior probability of taking each item (in scan order) when it still fits."""
    w = np.asarray(w, dtype=np.int64)
    n = len(w)
    remaining = int(C)
    greedy = np.zeros(n, dtype=bool)
    for i, weight in enumerate(w):
        if weight <= remaining:
            greedy[i] = True
            remaining -= int(weight)
    bias = n / 4
    return np.where(greedy, (1 + bias) / (2 + bias), 1 / (2 + bias))


# ---- energies (torch, float64 / complex128)


def _shift(values, w, C):
    """values[..., t] -> out[..., t + w] (t + w <= C)."""
    pad = torch.zeros(values.shape[:-1] + (w,), dtype=values.dtype)
    return torch.cat((pad, values[..., : C + 1 - w]), dim=-1)


def _overlaps(gammas, s, weights, C, S, q):
    """E_p[exp(i G phi)] for the phases G in gammas (J,) and rows s (R, n) -> (R, J)."""
    R = s.shape[0]
    A = torch.zeros((R, gammas.shape[0], C + 1), dtype=torch.complex128)
    A[:, :, 0] = 1.0
    t = torch.arange(C + 1)
    for i, w in enumerate(weights):
        can = (t <= C - w).to(torch.float64)
        f = torch.exp(1j * gammas[None, :, None] * s[:, i, None, None] / S)
        qi = q[i]
        A = A * (1 - can * qi) + _shift(A * can * f * qi, w, C)
    return A.sum(-1)


def _terms(angles, s, weights, C, S, q):
    """(Gamma_j, alpha_j), j = 0..L, of a(x) after the L layers in `angles`."""
    gammas = [torch.zeros((), dtype=torch.float64)]
    alphas = [torch.ones(s.shape[0], dtype=torch.complex128)]
    for layer in range(angles.shape[0] // 2):
        g, b = angles[2 * layer], angles[2 * layer + 1]
        gammas = [G + g for G in gammas]
        mu = _overlaps(torch.stack(gammas), s, weights, C, S, q)
        ep = (torch.stack(alphas, -1) * mu).sum(-1)
        gammas.append(torch.zeros((), dtype=torch.float64))
        alphas.append(-(1 - torch.exp(-1j * b)) * ep)
    return torch.stack(gammas), torch.stack(alphas, -1)


def _energies(angles, s, profits, weights, C, S, q):
    """-E[sum_i profits_i x_i] / S for rows s, profits (R, n) whose items are in scan order."""
    gam, alp = _terms(angles, s, weights, C, S, q)
    D = gam[None, :] - gam[:, None]
    R, J = s.shape[0], gam.shape[0]
    A = torch.zeros((R, J, J, C + 1), dtype=torch.complex128)
    A[..., 0] = 1.0
    B = torch.zeros_like(A)  # E[e^{i D phi} reward 1{t}]
    t = torch.arange(C + 1)
    for i, w in enumerate(weights):
        can = (t <= C - w).to(torch.float64)
        f = torch.exp(1j * D[None, :, :, None] * s[:, i, None, None, None] / S)
        p = profits[:, i, None, None, None]
        qi = q[i]
        keep = 1 - can * qi
        A, B = (A * keep + _shift(A * can * f * qi, w, C), B * keep + _shift((B + p * A) * can * f * qi, w, C))
    coef = alp.conj()[:, :, None] * alp[:, None, :]
    return -(coef * B.sum(-1)).sum((-2, -1)).real / S


def target(y):
    return y


def instance_energy(angles, coefficients, profits, spec):
    """Expected energy of one instance; coefficients (predicted) and profits are (n,)."""
    weights = [int(w) for w in np.asarray(spec.weights)]
    perm = scan_order(coefficients.numpy(), weights)  # the scan order depends on the predicted coefficients
    idx = torch.as_tensor(perm)
    w = [weights[i] for i in perm]
    q = [float(v) for v in take_probabilities(w, KP_CAPACITY)]
    return _energies(angles, coefficients[None, idx], profits[None, idx], w, KP_CAPACITY, spec.S, q)[0]


# ---- TEST shots


def _backward_messages(Gam, s, w, C, S, q):
    """R[k][t]: the (J, J) Gram matrix of items k.. given accumulated weight t, for the phases Gamma_j."""
    J = len(Gam)
    n = len(w)
    D = Gam[None, :] - Gam[:, None]
    R = np.zeros((n + 1, C + 1, J, J), complex)
    R[n, :] = 1.0
    for i in range(n - 1, -1, -1):
        wi = w[i]
        ph = np.exp(1j * D * s[i] / S)
        ps, pt = 1.0 - q[i], q[i]
        for t in range(C + 1):
            R[i, t] = (ps * R[i + 1, t] + pt * ph * R[i + 1, t + wi]) if t <= C - wi else R[i + 1, t]
    return R


def _draw(angles, s, w, C, S, shots, seed):
    """Shot counts of one instance, keyed by the item bits (item 0 first): the items are drawn one after another in
    scan order, each from its exact conditional distribution, with a NumPy PCG64 stream seeded by `seed`."""
    s = np.asarray(s, np.float64)
    w = np.asarray(w, dtype=np.int64)
    perm = scan_order(s, w)
    q = take_probabilities(w[perm], C)
    s, w = s[perm], w[perm]
    n = len(w)
    weights, prior = [int(v) for v in w], [float(v) for v in q]
    Gam, alp = _terms(torch.as_tensor(angles), torch.as_tensor(s)[None], weights, C, S, prior)
    Gam, alp = Gam.numpy(), alp[0].numpy()
    R = _backward_messages(Gam, s, w, C, S, q)
    coef = np.conj(alp)[:, None] * alp[None, :]
    D = Gam[None, :] - Gam[:, None]
    rng = np.random.default_rng(seed)
    t = np.zeros(shots, np.int64)
    phi_x = np.zeros(shots)
    x = np.zeros((shots, n), np.int8)

    def branch(i, b, tt):
        """Unnormalized probability of taking (b = 1) or skipping (b = 0) item i, at accumulated weight tt."""
        ph = np.exp(1j * D[None] * (phi_x + b * s[i] / S)[:, None, None])
        return np.maximum(np.real(np.einsum("ij,sij->s", coef, ph * R[i + 1, tt])), 0.0)

    for i, wi in enumerate(w):
        ps, pt = 1.0 - q[i], q[i]
        can = t <= C - wi
        w1 = np.where(can, pt * branch(i, 1, np.minimum(t + wi, C)), 0.0)
        skip = branch(i, 0, t)
        w0 = np.where(can, ps * skip, skip)
        take = can & (rng.random(shots) * (w0 + w1) < w1)
        x[:, i] = take
        phi_x += take * s[i] / S
        t += wi * take
    items = np.zeros_like(x)
    items[:, perm] = x  # scan order -> original item order
    counts = {}
    for row in items:
        k = "".join(str(int(b)) for b in row)
        counts[k] = counts.get(k, 0) + 1
    return counts


def sample_counts(params, X, spec, shots, seed):
    """Shot counts of every TEST instance, and the predicted coefficients they were drawn with.
    Instance r uses the stream seed + r."""
    angles, predicted = (tensor.numpy() for tensor in predict(params, X, spec, DEVICE))
    counts = [
        _draw(angles, predicted[r], spec.weights, KP_CAPACITY, spec.S, shots, seed + r) for r in range(len(predicted))
    ]
    return counts, predicted
