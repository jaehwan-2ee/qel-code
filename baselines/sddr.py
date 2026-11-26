"""SDDR baselines: what the training and TEST commands share."""

import numpy as np
import torch

from src.regret import relative_regrets


def cost_scale(problem, y_train):
    """Mean over the TRAIN instances of the summed absolute objective coefficients."""
    y = y_train.astype(np.float64)
    values = y.sum(-1) if problem == "ba" else y.reshape(len(y), -1)
    return float(np.abs(values).sum(-1).mean())


def require_device(settings):
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("this SDDR cell runs on a GPU; no CUDA GPU is visible")


def scores(model, x):
    """Scores on the CPU, where the rule is evaluated; the model may be on a GPU."""
    return model(x.to(next(model.parameters()).device)).cpu()


def decide(rule, scores, y, optimum):
    """(decisions, feasibility, relative regret) of the hard rule; an infeasible decision has relative regret 1."""
    decisions = rule.hard(scores).detach().cpu().numpy()
    feasible = rule.feasible(decisions)
    objective = -rule.cost(torch.tensor(decisions, dtype=torch.float64), y.double()).numpy()
    return decisions, feasible, np.where(feasible, relative_regrets(optimum, objective), 1.0)
