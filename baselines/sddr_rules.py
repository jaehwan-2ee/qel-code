"""The SDDR decision rule of each problem and its training loss.

A decision is read off the scores: the website with the largest score (BA), the items with a non-negative score (KP),
for every left node the right node with the largest score (BM). Training replaces it by a sigmoid with inverse
temperature eta and adds a penalty for scores that come within MARGIN of an infeasible decision:
loss = mean(cost / cost scale + penalty weight * penalty).
"""

import numpy as np
import torch

from data import BM_NODES, KP_CAPACITY

MARGIN = 0.3


def inverse_temperature(epoch, epochs):
    """eta of epoch 0, 1, ..., epochs - 1 of a fit with `epochs` epochs: it rises linearly from 1 towards 4."""
    return 1 + 3 * epoch / epochs


def choice_margins(scores):
    """Score of every choice minus the best other score of its group (last axis)."""
    n = scores.shape[-1]
    other = scores.unsqueeze(-2).expand(*scores.shape, n)
    other = other.masked_fill(torch.eye(n, dtype=torch.bool, device=scores.device), -torch.inf)
    return scores - other.amax(-1)


def minimal_cover_penalty(scores, weights, capacity):
    """KP penalty: the sum of relu(min of the scores in C + MARGIN) over all minimal covers C, i.e. item sets that
    exceed the capacity and stop doing so when any item is removed.

    The covers are not enumerated. With the items sorted by descending score, the minimum of a cover is the score of
    its last item, so it is enough to count, for every prefix, the covers that end there. A set with smallest weight m
    is a minimal cover iff capacity < total weight <= capacity + m; the dynamic program counts sets by (m, total).
    """
    ordered, order = torch.sort(scores, descending=True, stable=True, dim=-1)
    selected_weights = weights[order.detach().cpu().numpy()]
    batch, n = scores.shape
    maximum = int(weights.max())
    totals = int(capacity) + maximum + 1
    cutoffs = np.arange(1, maximum + 2)
    dp = np.zeros((batch, maximum + 1, totals), dtype=np.int64)
    dp[:, :, 0] = 1
    allowed_totals = (np.arange(totals)[None] > capacity) & (
        np.arange(totals)[None] <= capacity + np.arange(1, maximum + 1)[:, None]
    )
    previous = np.zeros(batch, dtype=np.int64)
    added = []
    for k in range(n):
        w = selected_weights[:, k]
        shifted_index = np.arange(totals)[None, :] - w[:, None]
        source = np.take_along_axis(dp, np.maximum(shifted_index, 0)[:, None, :], axis=2)
        dp += source * (shifted_index[:, None, :] >= 0) * (w[:, None, None] >= cutoffs[None, :, None])
        count = ((dp[:, :-1] - dp[:, 1:]) * allowed_totals).sum((1, 2))
        added.append(count - previous)
        previous = count
    coefficient = scores.new_tensor(np.stack(added, -1))
    return (coefficient * torch.relu(ordered + MARGIN)).sum(-1)


class Rule:
    def __init__(self, problem, weights=None):
        self.problem = problem
        self.weights = None if weights is None else np.asarray(weights, int)

    def _matrix(self, scores):
        return scores.reshape(-1, BM_NODES, BM_NODES)

    def penalty(self, scores):
        """KP: the minimal covers. BM: two left nodes whose best right node is, within the margin, the same one."""
        if self.problem == "kp":
            return minimal_cover_penalty(scores, self.weights, KP_CAPACITY)
        m = choice_margins(self._matrix(scores))
        a, b = torch.triu_indices(BM_NODES, BM_NODES, offset=1)
        return torch.relu(torch.minimum(m[:, a], m[:, b]) + MARGIN).sum((1, 2))

    def soft(self, scores, eta):
        if self.problem == "bm":
            return torch.sigmoid(eta * choice_margins(self._matrix(scores))).reshape(len(scores), -1)
        if self.problem == "ba":
            return torch.sigmoid(eta * choice_margins(scores))
        return torch.sigmoid(eta * scores)

    def hard(self, scores):
        if self.problem == "kp":
            return (scores >= 0).to(scores.dtype)
        grouped = scores if self.problem == "ba" else self._matrix(scores)
        chosen = torch.nn.functional.one_hot(grouped.argmax(-1), grouped.shape[-1])
        return chosen.reshape(len(scores), -1).to(scores.dtype)

    def cost(self, decisions, y):
        """Minus the objective of every instance under the true values y."""
        values = y.sum(-1) if self.problem == "ba" else y.reshape(len(y), -1)
        return -(decisions * values).sum(-1)

    def loss(self, scores, y, eta, penalty_weight, cost_scale):
        cost = self.cost(self.soft(scores, eta), y) / cost_scale
        if self.problem == "ba":  # choosing one website is always feasible: no penalty
            return cost.mean()
        return (cost + penalty_weight * self.penalty(scores)).mean()

    def feasible(self, decisions):
        z = np.asarray(decisions)
        if self.problem == "ba":
            return z.sum(-1) == 1
        if self.problem == "kp":
            return z @ self.weights <= KP_CAPACITY
        z = z.reshape(-1, BM_NODES, BM_NODES)
        return (z.sum(1) == 1).all(-1) & (z.sum(2) == 1).all(-1)
