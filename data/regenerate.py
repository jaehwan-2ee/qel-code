"""Regenerate the 15 datasets (BA, KP and BM for the data seeds 2025 to 2029) from the raw source files.

  RAW/ba/budget_allocation_data.pkl   RAW/kp/prices2013.dat   RAW/bm/cora.cites   RAW/bm/cora.content
  DATA/<problem>/<seed>/              X.npy, y.npy, weights.npy (KP only), meta.json

Every X.npy, y.npy and weights.npy is compared with its pinned sha256 (configs/data_sha256.json). The command ends
with an error if a file differs. A dataset that already exists in DATA is not generated again, only compared.
"""

import src  # noqa: F401, I001  first: it sets the thread environment before numpy and torch load

import argparse
from pathlib import Path

from data import PROBLEMS, ba_gen, bm_gen, kp_gen
from data.common import save_dataset
from src.files import read_json, sha256
from src.reported import CONFIGS, PAIRS

GENERATE = {"ba": ba_gen.generate, "kp": kp_gen.generate, "bm": bm_gen.generate}
SEEDS = [data_seed for data_seed, _ in PAIRS]


def regenerate(raw, data, problems, seeds):
    digests, differing = read_json(CONFIGS / "data_sha256.json"), []
    for problem in problems:
        for seed in seeds:
            target = data / problem / str(seed)
            if not target.exists():
                save_dataset(target, *GENERATE[problem](raw, seed))
            pinned = digests[problem][str(seed)].items()
            wrong = [
                name for name, digest in pinned if not (target / name).is_file() or sha256(target / name) != digest
            ]
            try:  # meta.json is written last
                read_json(target / "meta.json")
            except (OSError, ValueError):
                wrong.append("meta.json")
            print(problem, seed, "differs: " + ", ".join(wrong) if wrong else "identical", flush=True)
            differing += wrong
    if differing:
        raise SystemExit(f"{len(differing)} file(s) are missing or differ from configs/data_sha256.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m data.regenerate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, required=True, help="directory of the raw source files")
    parser.add_argument("--data", type=Path, required=True, help="directory of the datasets")
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS, help="default: all three")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, choices=SEEDS, help="default: all five")
    a = parser.parse_args()
    regenerate(a.raw, a.data, a.problems, a.seeds)
