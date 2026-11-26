"""From the shots of a TEST instance to one decision, and the TEST report.

The decision is the feasible bitstring that was sampled most often. Bitstrings that share the top count are ordered by
their objective under the predicted coefficients, and those still tied by the bitstring itself (the smallest one wins).
No true value enters the decision.
"""

import numpy as np

from data import BM_NODES, KP_CAPACITY
from src.regret import relative_regret, test_record


def _bits(bitstrings):
    """0/1 matrix of equal-length bitstrings, one row each."""
    return (np.frombuffer("".join(bitstrings).encode("ascii"), np.uint8).reshape(len(bitstrings), -1) - 48).astype(
        np.int64
    )


def feasible(problem, bitstring, weights):
    z = _bits([bitstring])[0]
    if problem == "ba":  # one website
        return z.sum() == 1
    if problem == "kp":
        return weights @ z <= KP_CAPACITY
    assignment = z.reshape(BM_NODES, BM_NODES)  # a perfect matching
    return bool((assignment.sum(0) == 1).all() and (assignment.sum(1) == 1).all())


def objective(problem, bitstring, y):
    """Objective of a decision under the true values y of its instance."""
    if problem == "ba":
        return float(np.asarray(y, dtype=float)[bitstring.index("1")].sum())
    return float(np.asarray(y, dtype=float).reshape(-1) @ np.fromiter(bitstring, dtype=np.float64))


def decide(problem, counts, predicted, weights):
    """The decision of a TEST instance from its shot counts, or None when no shot is feasible."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = [(bits, count) for bits, count in ordered if feasible(problem, bits, weights)]
    top = [bits for bits, count in top if count == top[0][1]]
    if not top:
        return None
    values = list(_bits(top) @ np.asarray(predicted, np.float64).reshape(-1))
    best = max(values)
    candidates = [bits for bits, value in zip(top, values) if value == best]
    return min(candidates)


def test_report(problem, counts, predicted, y, optimum, weights):
    """TEST record of the decisions; an instance without a feasible shot counts as relative regret 1."""
    decisions = [
        decide(problem, instance_counts, predicted[row], weights) for row, instance_counts in enumerate(counts)
    ]
    regrets = [
        1.0 if decision is None else relative_regret(optimum[row], objective(problem, decision, y[row]))
        for row, decision in enumerate(decisions)
    ]
    return test_record(regrets, optimum, sum(decision is None for decision in decisions))
