"""Adam with bias correction, on float32 torch tensors."""

import torch

BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8


def step(params, gradient, first, second, t, lr):
    """Update number t (from 1): returns the new parameters and the two moment estimates."""
    first = BETA1 * first + (1 - BETA1) * gradient
    second = BETA2 * second + (1 - BETA2) * torch.square(gradient)
    corrected_first = first / (1 - BETA1**t)
    corrected_second = second / (1 - BETA2**t)
    return params - lr * (corrected_first / (torch.sqrt(corrected_second) + EPSILON)), first, second
