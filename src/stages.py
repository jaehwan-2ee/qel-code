"""Stages 2 and 3 of a QEL fit: Adam on the SPSA gradient of the mean TRAIN energy, one update per epoch."""

import json
import math

import numpy as np
import torch

from src import adam
from src.early_stopping import early_stopping
from src.encoder import predict
from src.protocol import ANGLE_WARMUP_EPOCHS, JOINT_EPOCHS
from src.spsa import Gradient


def angle_warmup(circuit, X, y, spec, params, seed, lr):
    """Stage 2: the circuit angles are trained on the TRAIN energy with the encoder fixed. Returns the parameters."""
    gradient = Gradient(circuit, X, y, spec, seed)
    p = torch.tensor(params, dtype=torch.float32, device=circuit.DEVICE)
    n = gradient.n
    first = second = torch.zeros_like(p[n:])
    with torch.no_grad():
        for epoch in range(1, ANGLE_WARMUP_EPOCHS + 1):
            g = gradient.angles(p)
            p[n:], first, second = adam.step(p[n:], g, first, second, epoch, lr)
    return p.cpu().numpy()


def valid_energy(circuit, params, X, y, spec):
    """Mean expected energy of the VALID instances, computed exactly (no sampling)."""
    angles, predicted = predict(params, X, spec, circuit.DEVICE)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=circuit.DEVICE)
    energies = [circuit.instance_energy(angles, c, circuit.target(values), spec) for c, values in zip(predicted, y)]
    return float(torch.stack(energies).mean())


def joint_training(circuit, train, valid, spec, params, seed, lr, fit):
    """Stage 3: the encoder and the circuit angles are trained at one learning rate. After every epoch the VALID energy
    is evaluated; the checkpoint FIT/selected.npy is the best epoch on it under the early-stopping rule, and all epochs
    are run. Writes FIT/epochs.jsonl and FIT/epoch_params.npy; returns (VALID energies, best epoch, stop epoch)."""
    gradient = Gradient(circuit, *train, spec, seed)
    p = torch.tensor(params, dtype=torch.float32, device=circuit.DEVICE)
    first = second = torch.zeros_like(p)
    history = []
    epoch_params = np.zeros((JOINT_EPOCHS, len(p)), np.float32)
    with torch.no_grad(), (fit / "epochs.jsonl").open("w") as log:
        for epoch in range(1, JOINT_EPOCHS + 1):
            train_energy, g = gradient.joint(p)
            p, first, second = adam.step(p, g, first, second, epoch, lr)
            energy = valid_energy(circuit, p, *valid, spec)
            if not (math.isfinite(train_energy) and math.isfinite(energy)):
                raise FloatingPointError(f"non-finite energy at joint epoch {epoch}")
            history.append(energy)
            epoch_params[epoch - 1] = p.cpu().numpy()
            best, stop = early_stopping(history)
            if stop is None and best == epoch:
                np.save(fit / "selected.npy", epoch_params[epoch - 1])
            log.write(json.dumps(dict(epoch=epoch, train_energy=train_energy, valid_energy=energy)) + "\n")
            log.flush()
            print(f"epoch {epoch:3d}  TRAIN energy {train_energy:.6f}  VALID energy {energy:.6f}", flush=True)
    np.save(fit / "epoch_params.npy", epoch_params)
    return history, best, stop
