"""BA (budget allocation) dataset of one data seed, from the Yahoo! array `budget_allocation_data.pkl`.

The construction is that of OpenPTO's BudgetAllocation (_load_instances and _generate_features at commit f885426f)
with 5 websites and 10 users: two pools of instances are drawn with child seeds of the data seed, and the features of
a website are a random linear map of its standardized user values with ten noise columns appended. As there, the
first 80 instances of the 400-instance pool are VALID. The features are then standardized on the TRAIN rows.
"""

import pickle
import random

import numpy as np
import torch

from data import SPLITS
from data.common import check_raw_file, standardize_features

RAW_SIZE = 400_119_400
RAW_SHA256 = "11a0ac25f526c11f9fee9808c78f77055e036022540a5dd433a90e10393c786f"
N_WEBSITES, N_USERS = 5, 10


def _pool(Yfull, n_instances, seed):
    """Instances, websites and users drawn with replacement, as upstream does; returns (values, the drawn indices)."""
    np.random.seed(seed)
    indices = [np.random.choice(Yfull.shape[axis], n) for axis, n in enumerate((n_instances, N_WEBSITES, N_USERS))]
    Y = np.take(np.take(np.take(Yfull, indices[0], axis=0), indices[1], axis=1), indices[2], axis=2)
    return Y, dict(zip(("instances", "websites", "users"), (index.tolist() for index in indices)))


def _features(pools):
    """One random linear map 20 -> 10 for both pools, applied to [standardized user values | 10 noise columns]."""
    transform = torch.nn.Linear(2 * N_USERS, N_USERS)
    features = []
    for pool in pools:
        Y = torch.from_numpy(np.asarray(pool, dtype=np.float32))
        mean = Y.reshape(-1, N_USERS).mean(dim=0)
        std = Y.reshape(-1, N_USERS).std(dim=0)
        noise = torch.normal(mean=torch.zeros(Y.shape[0], Y.shape[1], N_USERS))
        features.append(transform(torch.cat(((Y - mean) / (std + 1e-10), noise), dim=2)).detach().clone().numpy())
    return features


def generate(raw, seed):
    """X (600, 5, 10) float32 and y (600, 5, 10), the user values; rows TRAIN 0:320, VALID 320:400, TEST 400:600."""
    path = raw / "ba/budget_allocation_data.pkl"
    check_raw_file(path, RAW_SIZE, RAW_SHA256)
    with path.open("rb") as handle:
        Yfull = np.asarray(pickle.load(handle)[0])
    n_train, valid_end, test_end = SPLITS["ba"]
    rng = random.Random(seed)
    child_seeds = {"training": rng.randrange(2**32), "test": rng.randrange(2**32)}
    (Y_train, drawn_train), (Y_test, drawn_test) = (
        _pool(Yfull, valid_end, child_seeds["training"]),
        _pool(Yfull, test_end - valid_end, child_seeds["test"]),
    )
    torch.manual_seed(child_seeds["test"])  # the state from which _features draws
    X_train, X_test = _features((Y_train, Y_test))
    n_valid = valid_end - n_train
    X, scaling = standardize_features(np.concatenate((X_train[n_valid:], X_train[:n_valid], X_test)), n_train)
    metadata = {
        "seed": seed,
        "child_seeds": child_seeds,
        "drawn_indices": {"training": drawn_train, "test": drawn_test},
        "feature_scaling": scaling,
    }
    return {"X": X, "y": np.concatenate((Y_train[n_valid:], Y_train[:n_valid], Y_test))}, metadata
