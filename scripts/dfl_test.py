"""TEST of one trained DFL baseline: OpenPTO's evaluation of the reported checkpoint FIT/selected.pt on the TEST rows.
Writes FIT/test.json (test_RR is the mean relative regret over the TEST instances) and FIT/test_arrays.npz
(predictions, decisions and relative regret of every TEST instance). KP and BM need a Gurobi license.

example:
  python -m scripts.dfl_test --fit OUT/dfl/fits/kp-h8-spo-d2025-m1020 --inputs INPUTS
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy and torch load


import numpy as np

from baselines import dfl
from data.split_inputs import load_test, load_train_valid
from src.files import write_json
from src.regret import relative_regrets, test_record
from src.reported import fit_identity, test_parser, trained_fit


def test(fit, inputs):
    fit = fit.resolve()  # OpenPTO runs in its own working directory
    selection = trained_fit(fit, "dfl", lambda s: dfl.SETTINGS)
    problem, seed = selection["problem"], selection["model_seed"]
    X, y, weights = load_train_valid(problem, selection["data_seed"], inputs)
    X_test, y_test, optimum, _ = load_test(problem, selection["data_seed"], inputs)
    arrays = np.concatenate((X, X_test)), np.concatenate((y, y_test)), weights
    # OpenPTO's loop with zero epochs: it evaluates the loaded model on TRAIN, VALID and TEST
    p, solver, loss, exp = dfl.start(
        problem, selection["method"], selection["hidden"], seed, arrays, fit / "openpto", fit / selection["checkpoint"]
    )
    final = dfl.observe_test()
    exp.run(p, loss, solver, n_epochs=0, do_debug=False)
    decisions = np.asarray(final["sols_hat"])
    objective, feasible = dfl.objective_and_feasibility(problem, decisions, y_test, weights)
    regrets = np.where(feasible, relative_regrets(optimum, objective), 1.0)  # an infeasible decision counts as 1
    predictions = final["preds"].detach().cpu().numpy()
    np.savez(fit / "test_arrays.npz", predictions=predictions, decisions=decisions, RR=regrets)
    write_json(fit / "test.json", fit_identity(selection) | test_record(regrets, optimum, (~feasible).sum()))
    dfl.finish(fit / "openpto")  # OpenPTO's working directory
    print(fit / "test.json")


if __name__ == "__main__":
    a = test_parser("dfl_test", __doc__).parse_args()
    test(a.fit, a.inputs)
