"""Train one reported DFL baseline with the training loop of OpenPTO: at most 300 epochs, early stopping on the VALID
regret with patience 50. Only the TRAIN and VALID rows are read. KP and BM need a Gurobi license.

Writes into FIT:
  epochs.jsonl     TRAIN loss and mean VALID regret of every epoch
  selected.pt      the reported checkpoint: the model of the best VALID epoch
  selection.json   the fit, its settings, the selected epoch and the sha256 of the checkpoint

example:
  python -m scripts.dfl_train --problem kp --method spo --hidden 8 --data-seed 2025 --model-seed 1020 \\
      --inputs INPUTS --fit OUT/dfl/fits/kp-h8-spo-d2025-m1020
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy and torch load


from baselines import dfl
from data.split_inputs import load_train_valid
from src.early_stopping import early_stopping
from src.files import sha256, write_json
from src.reported import DFL_METHODS, new_fit, train_parser


def train(problem, method, hidden, data_seed, model_seed, inputs, fit):
    new_fit(fit, data_seed, model_seed)
    fit = fit.resolve()  # OpenPTO runs in its own working directory
    arrays = load_train_valid(problem, data_seed, inputs)
    p, solver, loss, exp = dfl.start(problem, method, hidden, model_seed, arrays, fit / "openpto")
    history = dfl.observe_training(fit)
    exp.run(p, loss, solver, n_epochs=dfl.SETTINGS["max_epochs"], do_debug=False)
    best, stop = early_stopping(history)
    write_json(
        fit / "selection.json",
        dict(
            family="dfl",
            problem=problem,
            method=method,
            hidden=hidden,
            data_seed=data_seed,
            model_seed=model_seed,
            settings=dfl.SETTINGS,
            best_epoch=best,
            stop_epoch=stop,
            valid_regret=history[best - 1],
            checkpoint="selected.pt",
            checkpoint_sha256=sha256(fit / "selected.pt"),
        ),
    )
    dfl.finish(fit / "openpto")  # OpenPTO's working directory: its logs, checkpoints and the LODL samples


if __name__ == "__main__":
    parser = train_parser("dfl_train", __doc__)
    parser.add_argument("--method", choices=DFL_METHODS, required=True)
    a = parser.parse_args()
    train(a.problem, a.method, a.hidden, a.data_seed, a.model_seed, a.inputs, a.fit)
