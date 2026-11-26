"""The three datasets as OpenPTO problems.

The problem classes of OpenPTO load or generate their own data. The subclasses here are filled with the arrays of this
repository instead and keep the objective, the solver interface and the model shape of their parent, except where a
class says otherwise. The predictor is OpenPTO's dense MLP applied to every item: features of one item in, one
coefficient out.
"""

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from openpto.config import setup_seed
from openpto.method.Solvers.cvxpy.cp_bmatching import BmatchingSolver
from openpto.method.Solvers.grb.grb_knapsack import KPGrbSolver
from openpto.problems.BipartiteMatching import BipartiteMatching
from openpto.problems.BudgetAllocation import BudgetAllocation
from openpto.problems.Knapsack import Knapsack
from openpto.problems.PTOProblem import PTOProblem

from data import BM_NODES, KP_CAPACITY, SPLITS


class _Rows:
    """What the three problems share. X and y hold the TRAIN and VALID rows and, in the TEST process, the TEST rows."""

    def _start(self, problem, X, seed):
        PTOProblem.__init__(self)
        setup_seed(seed)  # every problem here uses the model seed
        self.has_test = len(X) > SPLITS[problem][1]
        return SPLITS[problem]

    def get_test_data(self, **kwargs):
        # The OpenPTO loop asks for a third partition before it trains. A training process has no TEST row and hands
        # out VALID again; the pass after training leaves that partition out (baselines/dfl.py).
        return super().get_test_data(**kwargs) if self.has_test else self.get_val_data(**kwargs)


class KnapsackSolver(KPGrbSolver):
    """OpenPTO's Gurobi knapsack solver; raises when Gurobi stops without an optimum."""

    def solve(self, y, **kwargs):
        result = super().solve(y, **kwargs)
        if self._model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi stopped with status {self._model.Status}")
        return result


class KnapsackProblem(_Rows, Knapsack):
    def __init__(self, X, y, weights, seed):
        n, valid_end, test_end = self._start("kp", X, seed)
        self.prob_version = "energy"
        self.num_items = X.shape[1]
        self.capacity = KP_CAPACITY
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.Xs_train, self.Xs_val, self.Xs_test = X[:n], X[n:valid_end], X[valid_end:]
        self.Ys_train, self.Ys_val, self.Ys_test = y[:n], y[n:valid_end], y[valid_end:]
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self.params_train = self.weights.unsqueeze(0).expand(n, -1)
        self.params_val = self.weights.unsqueeze(0).expand(valid_end - n, -1)
        self.params_test = self.weights.unsqueeze(0).expand(test_end - valid_end, -1)


def exact_assignment(Y):
    """The optimal assignment of every score matrix in Y, as 0/1 rows: a Gurobi integer program per instance.
    Gurobi takes its thread count from the gurobi.env of the working directory (baselines/dfl.py)."""
    n = BM_NODES
    scores = torch.as_tensor(Y).detach().cpu().double().reshape(-1, n * n).numpy()
    Z = np.zeros_like(scores)
    for k, profit in enumerate(scores):
        model = gp.Model("BM")
        model.setParam("OutputFlag", 0)
        x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        model.update()
        model.setObjective(
            gp.quicksum(float(profit[i * n + j]) * x[i][j] for i in range(n) for j in range(n)), GRB.MAXIMIZE
        )
        for i in range(n):
            model.addConstr(gp.quicksum(x[i][j] for j in range(n)) == 1, name=f"row_{i}")
        for j in range(n):
            model.addConstr(gp.quicksum(x[i][j] for i in range(n)) == 1, name=f"col_{j}")
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi stopped with status {model.Status}")
        Z[k] = [round(x[i][j].X) for i in range(n) for j in range(n)]
        model.dispose()
    return torch.from_numpy(Z).float()


class MatchingProblem(_Rows, BipartiteMatching):
    """Training losses use the relaxed matching of OpenPTO (a cvxpy layer), as upstream. The decisions that are
    evaluated on VALID and TEST, and the optima they are compared with, are exact assignments instead: `evaluating`
    is set to the partition while baselines/dfl.py evaluates it."""

    evaluating = None
    # OpenPTO's loop stores the relaxed optima in these two attributes; the setters discard them, and the getters
    # compute the exact assignments.
    z_val_opt = property(lambda self: exact_assignment(self.get_val_data()[1]), lambda self, value: None)
    z_test_opt = property(lambda self: exact_assignment(self.get_test_data()[1]), lambda self, value: None)

    def __init__(self, X, y, weights, seed):
        n, valid_end, test_end = self._start("bm", X, seed)
        self.num_nodes = BM_NODES
        self.Xs = torch.as_tensor(X, dtype=torch.float32)
        self.Ys = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.train_idxs = list(range(n))
        self.val_idxs = list(range(n, valid_end))
        self.test_idxs = list(range(valid_end, test_end))
        self.opt_train = BmatchingSolver()._getModel(isTrain=True, num_nodes=BM_NODES)
        self.opt_test = BmatchingSolver()._getModel(isTrain=False, num_nodes=BM_NODES)

    def init_API(self):
        return {**super().init_API(), "num_nodes": self.num_nodes}

    def get_decision(self, Y, params=None, ptoSolver=None, isTrain=False, **kwargs):
        # only the evaluation of OpenPTO (print_metrics) passes do_debug; the training losses do not
        if "do_debug" in kwargs and self.evaluating in ("val", "test"):
            Z = exact_assignment(Y)
            return Z, self.get_objective(Y, Z, params)
        return super().get_decision(Y, params, ptoSolver=ptoSolver, isTrain=isTrain, **kwargs)


class BudgetProblem(_Rows, BudgetAllocation):
    """BA as a linear problem over five websites, which replaces the submodular objective and the solver of the
    parent. The coefficient of a website is the sum of its ten user values (labels of shape (N, 5, 1)), the objective is
    the coefficient of the chosen website, and the decision is the website with the largest predicted coefficient,
    in training and in evaluation."""

    def __init__(self, X, y, weights, seed):
        n, valid_end, _ = self._start("ba", X, seed)
        self.num_targets = 10
        self.budget = 1
        X = torch.as_tensor(X, dtype=torch.float32)
        sums = torch.as_tensor(y, dtype=torch.float32).sum(-1, keepdim=True)
        # the parent keeps VALID in front of TRAIN in one array
        self.Xs_train = torch.cat((X[n:valid_end], X[:n]))
        self.Ys_train = torch.cat((sums[n:valid_end], sums[:n]))
        self.Xs_test = X[valid_end:]
        self.Ys_test = sums[valid_end:]
        self.val_idxs = range(valid_end - n)
        self.train_idxs = range(valid_end - n, valid_end)

    def get_model_shape(self):
        return self.Xs_train.shape[-1], 1

    def get_output_activation(self):
        return "identity"

    def get_objective(self, Y, Z, aux_data=None, **kwargs):
        # The graph is kept: blackbox and identity differentiate through z, NCE and LTR through the coefficients y.
        y = torch.as_tensor(Y)
        z = torch.as_tensor(Z)
        return (y.sum(-1) * z).sum(-1)

    def get_decision(self, Y, params=None, ptoSolver=None, **kwargs):
        scores = torch.as_tensor(Y).detach().cpu().double().sum(-1)  # (N, 5, 1) from the predictor
        chosen = np.argmax(scores.numpy(), axis=-1)  # ties keep the lowest index
        decision = torch.zeros(scores.shape, dtype=torch.float32)
        decision[np.arange(len(decision)), chosen] = 1
        return decision, self.get_objective(Y, decision)


PROBLEM = {"ba": BudgetProblem, "kp": KnapsackProblem, "bm": MatchingProblem}
