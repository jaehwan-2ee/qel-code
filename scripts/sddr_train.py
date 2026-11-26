"""Train one reported SDDR fit: --epochs epochs, each one Adam step on the full TRAIN set. Only the TRAIN and VALID
rows are read.

Writes into FIT:
  epochs.jsonl     of every epoch: train_loss before the step and valid_RR, the VALID relative regret, after it
  selected.pt      the reported checkpoint: the best epoch on the VALID relative regret under the early-stopping
                   rule (patience 50)
  selection.json   the fit, its settings, the selected epoch and the sha256 of the checkpoint
The settings are those of configs/sddr_settings.json.

example:
  python -m scripts.sddr_train --problem kp --structure shared --epochs 200 --hidden 8 --data-seed 2025 \\
      --model-seed 1020 --inputs INPUTS --fit OUT/sddr/fits/kp-h8-shared-e200-d2025-m1020
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy and torch load

import json

import torch

from baselines.sddr import cost_scale, decide, require_device, scores
from baselines.sddr_models import flatten_context, make_model
from baselines.sddr_rules import Rule, inverse_temperature
from data import SPLITS
from data.split_inputs import load_train_valid, optimum
from src.early_stopping import early_stopping
from src.files import sha256, write_json
from src.reported import SDDR_EPOCHS, SDDR_STRUCTURES, new_fit, sddr_settings, train_parser


def train(problem, structure, epochs, hidden, data_seed, model_seed, inputs, fit):
    new_fit(fit, data_seed, model_seed)
    settings = sddr_settings(problem, hidden, structure)
    require_device(settings)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    X, y, weights = load_train_valid(problem, data_seed, inputs)
    n = SPLITS[problem][0]
    valid_optimum = optimum(problem, y[n:], weights)
    scale = cost_scale(problem, y[:n]) if settings["scale_cost"] else 1.0
    x, y = torch.from_numpy(flatten_context(problem, X)), torch.from_numpy(y)

    model = make_model(structure, x.shape[1], hidden, y.shape[1], model_seed).to(settings["device"])
    rule = Rule(problem, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    history = []
    with (fit / "epochs.jsonl").open("w") as log:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            eta = inverse_temperature(epoch - 1, epochs)
            loss = rule.loss(scores(model, x[:n]), y[:n], eta, settings.get("penalty_weight"), scale)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TRAIN loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, _, regrets = decide(rule, scores(model, x[n:]), y[n:], valid_optimum)
            history.append(float(regrets.mean()))
            best, stop = early_stopping(history)
            if stop is None and best == epoch:
                torch.save(model.state_dict(), fit / "selected.pt")
            log.write(json.dumps({"epoch": epoch, "train_loss": loss.item(), "valid_RR": history[-1]}) + "\n")
    write_json(
        fit / "selection.json",
        dict(
            family="sddr",
            problem=problem,
            structure=structure,
            epochs=epochs,
            hidden=hidden,
            data_seed=data_seed,
            model_seed=model_seed,
            settings=settings,
            best_epoch=best,
            stop_epoch=stop,
            valid_RR=history[best - 1],
            checkpoint="selected.pt",
            checkpoint_sha256=sha256(fit / "selected.pt"),
        ),
    )


if __name__ == "__main__":
    parser = train_parser("sddr_train", __doc__)
    parser.add_argument(
        "--structure",
        choices=SDDR_STRUCTURES,
        required=True,
        help="independent: one network per decision variable; shared: one network",
    )
    parser.add_argument("--epochs", type=int, choices=SDDR_EPOCHS, required=True, help="number of epochs")
    a = parser.parse_args()
    train(a.problem, a.structure, a.epochs, a.hidden, a.data_seed, a.model_seed, a.inputs, a.fit)
