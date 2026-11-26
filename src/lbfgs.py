"""Stage 1, encoder warm-up: full-batch L-BFGS (float64, strong Wolfe line search) fits the encoder so that its output
matches the true coefficients of the TRAIN instances in mean squared error."""

import numpy as np
import torch

from src.encoder import coefficients
from src.protocol import LBFGS_ITERATIONS, true_coefficients

SETTINGS = dict(
    lr=1.0,
    max_iter=LBFGS_ITERATIONS,
    max_eval=5000,
    tolerance_grad=1e-8,
    tolerance_change=1e-12,
    history_size=20,
    line_search_fn="strong_wolfe",
)


def encoder_warmup(problem, hidden, X_train, y_train, initial, output_activation):
    """The fitted encoder parameters, float32. Runs on the CPU."""
    target = true_coefficients(problem, np.asarray(y_train, np.float32))
    x = torch.as_tensor(np.array(X_train, np.float32).reshape(-1, X_train.shape[-1]), dtype=torch.float64)
    truth = torch.as_tensor(target.reshape(-1), dtype=torch.float64)
    encoder = torch.nn.Parameter(torch.as_tensor(np.asarray(initial, np.float32)).to(dtype=torch.float64).clone())
    optimizer = torch.optim.LBFGS([encoder], **SETTINGS)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = ((coefficients(encoder, x, hidden, output_activation) - truth) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss in the encoder warm-up")
        loss.backward()
        return loss

    optimizer.step(closure)
    return encoder.detach().to(torch.float32).cpu().numpy().copy()
