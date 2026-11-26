"""TEST of one trained QEL fit: 2048 shots per TEST instance from the reported checkpoint FIT/selected.npy, decoded as
described in src/decoding.py. Writes FIT/test.json (test_RR is the mean relative regret over the TEST instances).

example:
  python -m scripts.qel_test --fit OUT/qel/fits/kp-l4-h8-d2025-m1020 --inputs INPUTS
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy, torch and jax load

import numpy as np
import torch

from data.split_inputs import load_test
from src.decoding import test_report
from src.device import require_gpu
from src.files import write_json
from src.protocol import CIRCUIT, SHOTS, Spec
from src.reported import fit_identity, qel_settings, test_parser, trained_fit


def test(fit, inputs):
    selection = trained_fit(fit, "qel", lambda s: qel_settings(s["problem"]))
    require_gpu()
    torch.set_num_threads(1)
    problem = selection["problem"]
    X, y, optimum, weights = load_test(problem, selection["data_seed"], inputs)
    spec = Spec(problem, selection["hidden"], selection["settings"]["output_activation"], selection["S"], weights)
    with torch.no_grad():  # the shots are drawn from the stream model seed + 1
        counts, predicted = CIRCUIT[problem].sample_counts(
            np.load(fit / selection["checkpoint"]), X, spec, SHOTS, selection["model_seed"] + 1
        )
    report = test_report(problem, counts, predicted, y, optimum, weights)
    write_json(fit / "test.json", fit_identity(selection) | report)
    print(fit / "test.json")


if __name__ == "__main__":
    a = test_parser("qel_test", __doc__).parse_args()
    test(a.fit, a.inputs)
