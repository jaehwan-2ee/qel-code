"""TEST of one trained SDDR fit: the hard decisions of the reported checkpoint FIT/selected.pt on the TEST rows.
Writes FIT/test.json (test_RR is the mean relative regret over the TEST instances) and FIT/test_arrays.npz (scores,
decisions and relative regret of every TEST instance).

example:
  python -m scripts.sddr_test --fit OUT/sddr/fits/kp-h8-shared-e200-d2025-m1020 --inputs INPUTS
"""

import src  # noqa: F401, I001  first: it sets the thread and GPU determinism environment before numpy and torch load

import numpy as np
import torch

from baselines.sddr import decide, require_device, scores
from baselines.sddr_models import flatten_context, make_model
from baselines.sddr_rules import Rule
from data.split_inputs import load_test
from src.files import write_json
from src.regret import test_record
from src.reported import fit_identity, sddr_settings, test_parser, trained_fit


def test(fit, inputs):
    selection = trained_fit(fit, "sddr", lambda s: sddr_settings(s["problem"], s["hidden"], s["structure"]))
    require_device(selection["settings"])
    torch.set_num_threads(1)
    problem, device = selection["problem"], selection["settings"]["device"]
    X, y, optimum, weights = load_test(problem, selection["data_seed"], inputs)
    x = torch.from_numpy(flatten_context(problem, X))
    model = make_model(selection["structure"], x.shape[1], selection["hidden"], y.shape[1], selection["model_seed"])
    model.to(device).load_state_dict(torch.load(fit / selection["checkpoint"], weights_only=True, map_location=device))
    with torch.no_grad():
        s = scores(model, x)
        decisions, feasible, regrets = decide(Rule(problem, weights), s, torch.from_numpy(y), optimum)
    np.savez(fit / "test_arrays.npz", scores=s.numpy(), decisions=decisions, RR=regrets)
    write_json(fit / "test.json", fit_identity(selection) | test_record(regrets, optimum, (~feasible).sum()))
    print(fit / "test.json")


if __name__ == "__main__":
    a = test_parser("sddr_test", __doc__).parse_args()
    test(a.fit, a.inputs)
