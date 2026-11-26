"""The encoder: one tanh MLP with one hidden layer, applied to every item of an instance.
Its parameters are stored as one vector [W1 (H x F, row by row) | b1 | w2 | b2].

`coefficients` evaluates it in float64 for the encoder warm-up (which differentiates it by autograd), the SPSA
gradient, the VALID energy and the TEST shots. Its output layer is a sum of products.
"""

import numpy as np
import torch


def n_encoder(feature_dim, hidden):
    """Number of encoder parameters."""
    return hidden * (feature_dim + 2) + 1


def _hidden(encoder, X, hidden):
    """(tanh hidden units of X, w2, b2)."""
    H, F = hidden, X.shape[-1]
    W1, b1 = encoder[: H * F].reshape(H, F), encoder[H * F : H * F + H]
    return torch.tanh(X @ W1.T + b1), encoder[H * F + H : H * F + 2 * H], encoder[-1]


def coefficients(encoder, X, hidden, output_activation):
    """Predicted coefficients of the items with features X (..., items, F). The output activation is the identity
    (BA, KP) or the sigmoid (BM)."""
    z, w2, b2 = _hidden(encoder, X, hidden)
    s = (z * w2).sum(-1) + b2
    return s.sigmoid() if output_activation == "sigmoid" else s


def jacobian_product(encoder, x, hidden, output_activation, upstream):
    """The gradient with respect to the encoder parameters of upstream . coefficients(encoder, x), for one instance x
    (items, F), written out by hand."""
    z, w2, _ = _hidden(encoder, x, hidden)
    u = upstream
    if output_activation == "sigmoid":
        s = coefficients(encoder, x, hidden, output_activation)
        u = u * s * (1 - s)
    pre = u[:, None] * w2[None, :] * (1 - z * z)
    return torch.cat(((pre.T @ x).reshape(-1), pre.sum(0), (u[:, None] * z).sum(0), u.sum().reshape(1)))


def predict(params, X, spec, device):
    """(circuit angles, predicted coefficients of the batch X) of a parameter vector, float64 tensors on `device`."""
    p = torch.as_tensor(params, dtype=torch.float64, device=device)
    X = torch.as_tensor(np.asarray(X, np.float64), device=device)
    n = n_encoder(X.shape[-1], spec.hidden)
    return p[n:], coefficients(p[:n], X, spec.hidden, spec.output_activation)
