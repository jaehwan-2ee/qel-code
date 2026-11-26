"""Train one reported QEL fit. Only the TRAIN and VALID rows are read; VALID enters through its exact energy.

Writes into FIT:
  start.npy          parameters before training: [encoder | circuit angles]
  joint_start.npy    parameters after the encoder warm-up and the angle warm-up
  epochs.jsonl       of every joint epoch: train_energy, the mean of the SPSA-perturbed TRAIN energies before the
                     update, and valid_energy, the VALID energy after it
  epoch_params.npy   parameters after every joint epoch
  selected.npy       the reported checkpoint: the best epoch on the VALID energy under the early-stopping rule
                     (patience 50)
  selection.json     the fit, its settings, the selected epoch and the sha256 of the checkpoint
The stages are described in src/protocol.py, the settings are those of configs/qel_settings.json.

example:
  python -m scripts.qel_train --problem kp --layers 4 --hidden 8 --data-seed 2025 --model-seed 1020 \\
      --inputs INPUTS --fit OUT/qel/fits/kp-l4-h8-d2025-m1020
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy, torch and jax load

import numpy as np
import torch

from data import SPLITS
from data.split_inputs import load_train_valid
from src.device import require_gpu
from src.encoder import n_encoder
from src.files import sha256, write_json
from src.lbfgs import encoder_warmup
from src.protocol import CIRCUIT, Spec, initial_parameters, scale
from src.reported import QEL_LAYERS, new_fit, qel_settings, train_parser
from src.stages import angle_warmup, joint_training


def train(problem, layers, hidden, data_seed, model_seed, inputs, fit):
    new_fit(fit, data_seed, model_seed)
    require_gpu()
    torch.set_num_threads(1)
    settings = qel_settings(problem)
    X, y, weights = load_train_valid(problem, data_seed, inputs)
    n = SPLITS[problem][0]
    train_rows, valid_rows = (X[:n], y[:n]), (X[n:], y[n:])
    spec = Spec(problem, hidden, settings["output_activation"], scale(problem, y[:n]), weights)
    circuit = CIRCUIT[problem]

    params = initial_parameters(problem, X.shape[-1], hidden, layers, model_seed, settings["output_scale"])
    np.save(fit / "start.npy", params)
    encoder = slice(0, n_encoder(X.shape[-1], hidden))
    params[encoder] = encoder_warmup(problem, hidden, *train_rows, params[encoder], settings["output_activation"])
    params = angle_warmup(circuit, *train_rows, spec, params, model_seed, settings["lr_angle_warmup"])
    np.save(fit / "joint_start.npy", params)
    history, best, stop = joint_training(
        circuit, train_rows, valid_rows, spec, params, model_seed, settings["lr_joint"], fit
    )
    write_json(
        fit / "selection.json",
        dict(
            family="qel",
            problem=problem,
            layers=layers,
            hidden=hidden,
            data_seed=data_seed,
            model_seed=model_seed,
            settings=settings,
            S=spec.S,
            best_epoch=best,
            stop_epoch=stop,
            valid_energy=history[best - 1],
            checkpoint="selected.npy",
            checkpoint_sha256=sha256(fit / "selected.npy"),
        ),
    )


if __name__ == "__main__":
    parser = train_parser("qel_train", __doc__)
    parser.add_argument("--layers", type=int, choices=QEL_LAYERS, required=True, help="circuit depth p")
    a = parser.parse_args()
    train(a.problem, a.layers, a.hidden, a.data_seed, a.model_seed, a.inputs, a.fit)
