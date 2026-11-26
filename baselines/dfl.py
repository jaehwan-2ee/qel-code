"""The eight DFL baselines, trained and evaluated by OpenPTO (third_party/predictiveco).

From OpenPTO: the training loop (ExpManager: Adam, one step per epoch on the per-instance losses, early stopping on the
VALID regret), the eight loss classes, the dense MLP, the KP Gurobi solver and the BM cvxpy layer.
From this repository: the data and the problem classes (baselines/dfl_problems.py), and this module, which starts that
loop and reads its results.

Settings that differ from the benchmark scripts of OpenPTO: batch size 1 for every method (theirs: 256 for blackbox and
identity, and for SPO on BA and KP); one hidden layer of 8 or 12 tanh units (their default: 32 ReLU units); patience 50
(their default: 40); the LODL configuration lodl5k.yaml on all three problems (their KP script uses default.yaml).
"""

import importlib
import json
import logging
import os
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from openpto.config import load_conf
from openpto.method.Models.wrapper_loss import get_loss_fn
from openpto.method.Solvers.wrapper_solver import solver_wrapper

from baselines.dfl_problems import PROBLEM, KnapsackSolver
from data import BM_NODES, KP_CAPACITY
from src.early_stopping import PATIENCE

# the module, which `import openpto.expmanager.ExpManager` would not give: the package exports the class by that name
manager_module = importlib.import_module("openpto.expmanager.ExpManager")
CONFIG = Path(manager_module.__file__).resolve().parents[1] / "config"
SETTINGS = {"lr": 0.01, "batch_size": 1, "max_epochs": 300, "patience": PATIENCE}  # recorded with every fit
OPENPTO_PROBLEM = {  # their problem name, problem configuration and solver
    "ba": ("budgetalloc", "budgetalloc.yaml", "neural"),
    "kp": ("knapsack", "knapsack-real.yaml", "gurobi"),
    "bm": ("bipartitematching", "bipartitematching.yaml", "cvxpy"),
}
OPENPTO_METHOD = {"point-ltr": "pointLTR", "pair-ltr": "pairLTR", "list-ltr": "listLTR"}  # other names are the same


def start(problem, method, hidden, model_seed, arrays, work, checkpoint=None):
    """Build what one run of OpenPTO's loop needs: (problem object, solver, loss, ExpManager).
    Training: the loss of `method`. TEST: the model is loaded from `checkpoint`, and the loop runs zero epochs.
    `work` becomes the working directory; OpenPTO writes its own logs and checkpoints there."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    shutil.rmtree(work, ignore_errors=True)  # an empty working directory, also when an unfinished fit left one
    work.mkdir(parents=True)
    os.chdir(work)
    Path("gurobi.env").write_text("Threads 1\n")  # Gurobi reads it from the working directory
    name, problem_file, solver_name = OPENPTO_PROBLEM[problem]
    method_file = "lodl/lodl5k.yaml" if method == "lodl" else "default.yaml"
    conf = load_conf(str(CONFIG / "probs" / problem_file), str(CONFIG / "models" / method_file), name)
    # the arguments of OpenPTO's benchmark scripts, except batch_size, n_hidden, activation, patience and gpu
    args = SimpleNamespace(
        problem=name,
        opt_model=OPENPTO_METHOD.get(method, method),
        solver=solver_name,
        pred_model="dense",
        gpu="-1",
        batch_size=SETTINGS["batch_size"],
        n_ptr_epochs=0,
        trained_path=str(checkpoint or ""),
        loss_path="",
        log_dir="records",
        bkup_log_dir="records-backup",
        lr=SETTINGS["lr"],
        use_lr_scheduling=False,
        n_layers=2,
        n_hidden=hidden,
        activation="tanh",
        earlystopping=True,
        patience=PATIENCE,
        valfreq=1,
        opt_name="gd",
    )
    for directory in (args.log_dir, args.bkup_log_dir):
        Path(directory, "checkpoints").mkdir(parents=True)
    p = PROBLEM[problem](*arrays, model_seed)
    p.device = torch.device("cpu")
    solver = solver_wrapper(args, conf, p)
    if problem == "kp":
        solver.__class__ = KnapsackSolver  # the same solver, with a check of the Gurobi status
    if checkpoint:
        loss = lambda problem, coeff_hat, **kwargs: torch.zeros(1)  # noqa: E731  the TEST pass needs no training loss
    else:
        if method == "lodl":
            seed_lodl(model_seed)
        loss = get_loss_fn(args, solver, conf)
    logger = logging.getLogger("openpto")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("log.txt"))
    logger.addHandler(logging.StreamHandler())
    shape = p.get_model_shape()
    model = {"ipdim": shape[0], "opdim": shape[1], "out_act": p.get_output_activation()}
    return p, solver, loss, manager_module.ExpManager(model, args, conf, logger)


def finish(work):
    """Close OpenPTO's log file, leave its working directory and remove it."""
    logger = logging.getLogger("openpto")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    os.chdir(work.parent)
    shutil.rmtree(work)


def seed_lodl(model_seed):
    """The LODL code of OpenPTO calls random.seed(0) before it samples the indices of its samples and random.seed()
    afterwards. Both calls are redirected to the model seed, so that the sampled indices depend on it."""
    seed = random.seed
    random.seed = lambda value=None, version=2: seed(model_seed, version=version)


def _evaluate(original, datasets, model, *rest, **kwargs):
    """OpenPTO's print_metrics, one partition at a time so that the BM problem knows which one it is evaluating."""
    problem, metrics = rest[0], {}
    for dataset in datasets:
        problem.evaluating = dataset[3]
        metrics.update(original([dataset], model, *rest, **kwargs))
    problem.evaluating = None
    return metrics


def observe_training(fit):
    """Wrap OpenPTO's print_metrics, which its loop calls after every epoch and once after training. Every epoch is
    appended to FIT/epochs.jsonl and its mean VALID regret to the returned list. After training, the model the loop
    ends with (its best VALID epoch) is saved as FIT/selected.pt."""
    original, history = manager_module.print_metrics, []

    def observed(datasets, model, problem, loss_fn, criterion, solver, prefix, logger, **kwargs):
        final = prefix == "Final"  # "Iter <epoch>," during training
        rest = problem, loss_fn, criterion, solver, prefix, logger
        metrics = _evaluate(original, datasets[:2] if final else datasets, model, *rest, **kwargs)
        if final:
            torch.save(model.state_dict(), fit / "selected.pt")
            metrics["test"] = metrics["val"]  # the loop reads a third entry for its closing log lines
            return metrics
        history.append(float(np.asarray(metrics["val"]["eval"]["value"]).mean()))
        row = {"epoch": len(history), "train_loss": metrics["train"]["loss"], "valid_regret": history[-1]}
        with (fit / "epochs.jsonl").open("a" if len(history) > 1 else "w") as log:
            log.write(json.dumps(row) + "\n")
        return metrics

    manager_module.print_metrics = observed
    return history


def observe_test():
    """Wrap print_metrics for a TEST process; returns the dict that receives the TEST metrics of OpenPTO."""
    original, final = manager_module.print_metrics, {}

    def observed(datasets, model, problem, loss_fn, criterion, solver, prefix, logger, **kwargs):
        rest = problem, loss_fn, criterion, solver, prefix, logger
        metrics = _evaluate(original, datasets, model, *rest, **kwargs)
        final.update(metrics["test"])
        return metrics

    manager_module.print_metrics = observed
    return final


def objective_and_feasibility(problem, decisions, y, weights):
    """(objective under the true values y, feasibility) of TEST decisions. The KP decisions are Gurobi solutions, hence
    the tolerances."""
    z = np.asarray(decisions, dtype=np.float64)
    y = y.astype(np.float64)
    if problem == "ba":
        y, z = y.sum(-1), z.reshape(len(y), -1)
    tolerance = 1e-5
    binary = ((np.abs(z) < tolerance) | (np.abs(z - 1) < tolerance)).all(-1)
    if problem == "ba":
        feasible = binary & (np.abs(z.sum(-1) - 1) < 1e-4)
    elif problem == "kp":
        feasible = binary & (z @ weights.astype(float) <= KP_CAPACITY + 1e-6)
    else:
        m = z.reshape(len(z), BM_NODES, BM_NODES)
        feasible = binary & (np.abs(m.sum(1) - 1) < 1e-4).all(-1) & (np.abs(m.sum(2) - 1) < 1e-4).all(-1)
    return (y * z).sum(-1), feasible
