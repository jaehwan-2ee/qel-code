"""QEL needs a CUDA GPU: the initial circuit angles are drawn there, and the BM circuit is simulated there."""

import torch


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("QEL needs a CUDA GPU; none is visible")
