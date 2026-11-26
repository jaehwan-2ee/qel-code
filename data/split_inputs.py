"""The split inputs of one (problem, data seed): the rows of a dataset, cut into the two parts a process may read.

  train-valid  INPUTS/<problem>/<seed>/X.npy, y.npy        the TRAIN and VALID rows, and weights.npy for KP
  test         INPUTS/<problem>/<seed>/test/X.npy, y.npy   the TEST rows, and for BA and KP optimum.npy, their optimum

Every training command reads the first part only, so training can run before the TEST part exists. Both parts are
checked against configs/split_inputs_sha256.json when they are written and whenever they are read.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data import BM_NODES, KP_CAPACITY, PROBLEMS, SPLITS
from src.files import read_json, sha256

DIGESTS = Path(__file__).resolve().parents[1] / "configs/split_inputs_sha256.json"


def optimum(problem, y, weights=None):
    """Optimal objective of every row, float64: the best website (BA), 0/1 dynamic programming (KP), linear sum
    assignment (BM)."""
    y = np.asarray(y, dtype=np.float64)
    if problem == "ba":
        return y.sum(-1).max(-1)
    if problem == "bm":
        matrices = y.reshape(-1, BM_NODES, BM_NODES)
        return np.asarray([m[linear_sum_assignment(m, maximize=True)].sum() for m in matrices])
    best = np.zeros((len(y), KP_CAPACITY + 1), dtype=np.float64)  # best[:, c]: best value within capacity c
    for item, weight in enumerate(int(w) for w in weights):
        best[:, weight:] = np.maximum(best[:, weight:], best[:, :-weight] + y[:, item, None])
    return best[:, -1]


def _digests(problem, data_seed, part):
    listed = read_json(DIGESTS)[problem]
    if str(data_seed) not in listed:
        raise SystemExit(f"{data_seed} is not a reported data seed; use one of {', '.join(listed)}")
    return listed[str(data_seed)][part]


def _check(directory, digests):
    for name, digest in digests.items():
        if not (directory / name).is_file():
            part = "test" if directory.name == "test" else "train-valid"
            raise SystemExit(f"{directory / name} is missing; write it with python -m data.split_inputs {part}")
        if sha256(directory / name) != digest:
            raise SystemExit(f"{directory / name} differs from configs/split_inputs_sha256.json")


def _rows(path, start, end):
    """Rows start:end of an array file; the other rows are not read."""
    if not path.is_file():
        raise SystemExit(f"{path} is missing; regenerate the datasets with python -m data.regenerate")
    return np.ascontiguousarray(np.load(path, mmap_mode="r")[start:end])


def _separate(source, out):
    if out.resolve() == source.resolve():
        raise SystemExit(f"{out} is the dataset directory itself; give --inputs another directory than --data")


def write_train_valid(problem, data_seed, data, inputs):
    digests = _digests(problem, data_seed, "train_valid")
    source, out = Path(data) / problem / str(data_seed), Path(inputs) / problem / str(data_seed)
    _separate(source, out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("X.npy", "y.npy"):
        np.save(out / name, _rows(source / name, 0, SPLITS[problem][1]))
    if problem == "kp":
        np.save(out / "weights.npy", np.load(source / "weights.npy"))
    _check(out, digests)
    return out


def write_test(problem, data_seed, data, inputs):
    digests = _digests(problem, data_seed, "test")
    source, out = Path(data) / problem / str(data_seed), Path(inputs) / problem / str(data_seed) / "test"
    _separate(source, out)
    out.mkdir(parents=True, exist_ok=True)
    _, valid_end, test_end = SPLITS[problem]
    y = _rows(source / "y.npy", valid_end, test_end)
    np.save(out / "X.npy", _rows(source / "X.npy", valid_end, test_end))
    np.save(out / "y.npy", y)
    if problem != "bm":
        weights = np.load(source / "weights.npy") if problem == "kp" else None
        np.save(out / "optimum.npy", optimum(problem, y, weights))
    _check(out, digests)
    return out


def load_train_valid(problem, data_seed, inputs):
    """(X, y, KP weights or None) of the TRAIN and VALID rows. No TEST file is opened."""
    directory = Path(inputs) / problem / str(data_seed)
    _check(directory, _digests(problem, data_seed, "train_valid"))
    weights = np.load(directory / "weights.npy") if problem == "kp" else None
    return np.load(directory / "X.npy"), np.load(directory / "y.npy"), weights


def load_test(problem, data_seed, inputs):
    """(X, y, optimum, KP weights or None) of the TEST rows. The BM optimum is solved here."""
    directory = Path(inputs) / problem / str(data_seed)
    _check(directory / "test", _digests(problem, data_seed, "test"))
    y = np.load(directory / "test/y.npy")
    weights = None
    if problem == "kp":
        _check(directory, {"weights.npy": _digests(problem, data_seed, "train_valid")["weights.npy"]})
        weights = np.load(directory / "weights.npy")
    best = optimum("bm", y) if problem == "bm" else np.load(directory / "test/optimum.npy")
    return np.load(directory / "test/X.npy"), y, best, weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m data.split_inputs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("part", choices=("train-valid", "test"))
    parser.add_argument("--problem", choices=PROBLEMS, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True, help="datasets written by data.regenerate")
    parser.add_argument("--inputs", type=Path, required=True, help="directory of the split inputs")
    a = parser.parse_args()
    write = write_train_valid if a.part == "train-valid" else write_test
    print(write(a.problem, a.data_seed, a.data, a.inputs))
