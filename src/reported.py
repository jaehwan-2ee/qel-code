"""The reported fits: the seed pairs, the cells of the table and the settings of every method.
Training and TEST refuse any other seed pair, and TEST refuses a fit that was trained with other settings."""

import argparse
from pathlib import Path

from data import PROBLEMS
from src.files import read_json, sha256

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PAIRS = [(2025, 1020), (2026, 1021), (2027, 1022), (2028, 1023), (2029, 1024)]  # (data seed, model seed)
HIDDEN = (8, 12)  # hidden width of every predictor
QEL_LAYERS = (4, 6)
SDDR_STRUCTURES = ("independent", "shared")
SDDR_EPOCHS = (200, 400, 600, 800, 1000)  # epoch budgets of SDDR, one fit per budget
DFL_METHODS = ("blackbox", "identity", "spo", "nce", "point-ltr", "pair-ltr", "list-ltr", "lodl")
CHOICES = {"qel": ("layers", QEL_LAYERS), "sddr": ("epochs", SDDR_EPOCHS)}  # the field of a fit that names its cell


def check_pair(data_seed, model_seed):
    if (data_seed, model_seed) not in PAIRS:
        pairs = ", ".join(f"({d}, {m})" for d, m in PAIRS)
        raise SystemExit(f"({data_seed}, {model_seed}) is not a reported (data seed, model seed) pair; use {pairs}")


def sddr_settings(problem, hidden, structure):
    """lr (Adam), scale_cost (divide the cost by the TRAIN cost scale), penalty_weight (KP and BM) and device (cpu
    or cuda) of one cell."""
    return read_json(CONFIGS / "sddr_settings.json")[f"{problem}-h{hidden}-{structure}"]


def fit_identity(selection):
    """The fields of selection.json that name the fit, with the selected epoch; they open its test.json."""
    keys = [k for k in selection if k in ("family", "problem", "layers", "structure", "epochs", "method", "hidden")]
    return {k: selection[k] for k in (*keys, "data_seed", "model_seed", "best_epoch", "stop_epoch")}


def qel_settings(problem):
    """lr_angle_warmup and lr_joint (Adam learning rates of stages 2 and 3), output_activation of the encoder, and
    output_scale (factor on the output layer of the encoder at the start)."""
    return read_json(CONFIGS / "qel_settings.json")[problem]


def train_parser(name, description):
    """Arguments shared by the three training commands."""
    parser = argparse.ArgumentParser(
        prog=f"python -m scripts.{name}", description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problem", choices=PROBLEMS, required=True)
    parser.add_argument("--hidden", type=int, choices=HIDDEN, required=True, help="hidden width of the predictor")
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="split inputs written by data.split_inputs")
    parser.add_argument("--fit", type=Path, required=True, help="directory of the fit; it is created")
    return parser


def test_parser(name, description):
    """Arguments of the three TEST commands."""
    parser = argparse.ArgumentParser(
        prog=f"python -m scripts.{name}", description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fit", type=Path, required=True, help="directory of a trained fit")
    parser.add_argument("--inputs", type=Path, required=True, help="split inputs written by data.split_inputs")
    return parser


def new_fit(fit, data_seed, model_seed):
    """Directory for a fit to train. A directory without selection.json is an unfinished fit and is trained again."""
    check_pair(data_seed, model_seed)
    if (fit / "selection.json").exists():
        raise SystemExit(f"{fit} is already trained; choose another --fit or delete the directory")
    fit.mkdir(parents=True, exist_ok=True)


def trained_fit(fit, family, settings_of):
    """selection.json of a fit to test. `settings_of(selection)` returns the reported settings of its cell."""
    if not (fit / "selection.json").is_file():
        raise SystemExit(f"{fit} has no selection.json; train the fit first")
    selection = read_json(fit / "selection.json")
    if selection["family"] != family:
        raise SystemExit(f"{fit} is a {selection['family']} fit; test it with scripts.{selection['family']}_test")
    check_pair(selection["data_seed"], selection["model_seed"])
    if family in CHOICES:
        key, allowed = CHOICES[family]
        if selection.get(key) not in allowed:
            raise SystemExit(f"{fit}: {key} {selection.get(key)} is not one of {', '.join(map(str, allowed))}")
    if selection["settings"] != settings_of(selection):
        raise SystemExit(f"{fit} was not trained with the reported settings")
    checkpoint = fit / selection["checkpoint"]
    if not checkpoint.is_file() or sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise SystemExit(f"{checkpoint} is not the checkpoint of selection.json; delete {fit} and train it again")
    if (fit / "test.json").exists():
        raise SystemExit(f"{fit} already has test.json; a fit is tested once")
    return selection
