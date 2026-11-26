"""Relative regret of a decision in a maximization problem: (optimum - value) / |optimum|."""

import math

import numpy as np


def relative_regret(optimum, value):
    """With a zero optimum the regret is 0 when the decision attains it and infinite otherwise. BA has such TEST
    instances (no website has a user value); every feasible decision attains their optimum."""
    optimum, value = float(optimum), float(value)
    if optimum == 0:
        return 0.0 if value == optimum else math.inf
    return (optimum - value) / abs(optimum)


def relative_regrets(optima, values):
    """relative_regret of every pair, as a float64 array."""
    return np.asarray([relative_regret(o, v) for o, v in zip(np.ravel(optima), np.ravel(values))])


def test_record(regrets, optima, infeasible_count):
    """The TEST fields that every method records. test_RR is the mean relative regret over the TEST instances."""
    return dict(
        test_RR=float(np.mean(regrets)),
        test_count=len(regrets),
        infeasible_count=int(infeasible_count),
        zero_optimum_count=int(sum(float(optimum) == 0 for optimum in optima)),
        test_RR_per_instance=[float(regret) for regret in regrets],
    )
