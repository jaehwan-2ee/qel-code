"""KP (energy knapsack) dataset of one data seed, from the price table `prices2013.dat`.

The table is cleaned as OpenPTO's Knapsack.get_energy_pandas does (commit f885426f): the same columns are dropped, zero
CO2 intensities are interpolated, and the three days with missing values are removed, which leaves 789 days of 48
half-hour slots. An instance is one day, an item a slot, its value the price SMPEP2. The 48 item weights are those of
Knapsack.get_energy_data. Upstream draws TRAIN and VALID from the first 552 days and tests on the rest; here TRAIN and
VALID are a seeded permutation of the first 400 days and TEST is the days 552:752, the same for every data seed. The
features are standardized on the TRAIN rows.
"""

import numpy as np
import pandas as pd

from data import SPLITS
from data.common import check_raw_file, standardize_features

RAW_SIZE = 4_455_611
RAW_SHA256 = "c8086b6b026c1edce01bea265392e42ce46245e9f7f5e88d871232e8c6176faf"
FEATURES = (
    "HolidayFlag",
    "DayOfWeek",
    "WeekOfYear",
    "Month",
    "ForecastWindProduction",
    "SystemLoadEA",
    "SMPEA",
    "CO2Intensity",
)
# fmt: off
WEIGHTS = np.array([
    5, 3, 3, 5, 5, 7, 7, 3, 7, 7, 3, 3,
    5, 3, 7, 3, 7, 7, 5, 5, 3, 5, 5, 3,
    7, 7, 3, 7, 5, 5, 7, 3, 7, 3, 3, 5,
    7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 3, 7,
], dtype=np.int64)
# fmt: on
SLOTS = len(WEIGHTS)
DROPPED = ("#DateTime", "Holiday", "ActualWindProduction", "SystemLoadEP2", "ORKTemperature", "ORKWindspeed")
FIRST_TEST_DAY = 552  # where upstream's test days begin: int(0.7 * 789)


def _complete_days(frame):
    clean = frame.drop(columns=list(DROPPED)).copy()
    clean["CO2Intensity"] = clean["CO2Intensity"].replace(0, np.nan).interpolate()
    clean["groupID"] = np.arange(len(clean)) // SLOTS
    complete = clean.groupby("groupID").filter(lambda day: not day.isna().any().any())
    return complete.drop(columns=["Day", "Year", "PeriodOfDay"])


def generate(raw, seed):
    """X (600, 48, 8) float32, y (600, 48) float32 and the 48 item weights; rows TRAIN 0:320, VALID 320:400,
    TEST 400:600."""
    path = raw / "kp/prices2013.dat"
    check_raw_file(path, RAW_SIZE, RAW_SHA256)
    clean = _complete_days(pd.read_csv(path, sep=r"\s+", quotechar='"'))
    X_days = clean.loc[:, FEATURES].to_numpy(dtype=np.float64).reshape(-1, SLOTS, len(FEATURES))
    y_days = clean["SMPEP2"].to_numpy(dtype=np.float64).reshape(-1, SLOTS)
    n_train, valid_end, test_end = SPLITS["kp"]
    permutation = np.random.RandomState(seed).permutation(valid_end)
    days = {
        "training": permutation[valid_end - n_train :],
        "validation": permutation[: valid_end - n_train],
        "test": np.arange(FIRST_TEST_DAY, FIRST_TEST_DAY + test_end - valid_end),
    }
    source = np.concatenate(list(days.values()))
    X, scaling = standardize_features(X_days[source], n_train)
    metadata = {
        "seed": seed,
        "source_days": {name: value.tolist() for name, value in days.items()},
        "feature_scaling": scaling,
    }
    return {"X": X, "y": y_days[source].astype(np.float32), "weights": WEIGHTS.copy()}, metadata
