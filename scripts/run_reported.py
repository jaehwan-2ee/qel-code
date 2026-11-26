"""Train and test the reported fits of one family: qel (60 fits), sddr (300) or dfl (240).

Every fit is trained by scripts.<family>_train and then tested by scripts.<family>_test, each in a process of its own.
The split inputs are written to OUT/<family>/inputs: the TRAIN/VALID part before the first fit starts, the TEST part
of a dataset when the first of its fits has been trained. A fit that has its test.json is skipped, and a fit without
selection.json is trained from the start, so the same command resumes a run. Ctrl-C ends the running fits and starts
no further one. Use one runner per family and output directory at a time.

  OUT/<family>/fits/<fit>/       selection.json, the checkpoint, test.json (see the training and TEST commands)
  OUT/<family>/logs/<fit>.log    the two commands and their output

<fit> is <problem>-l<layers>-h<hidden>-d<data seed>-m<model seed> for qel, with <structure>-e<epochs> (sddr) or
<method> (dfl) in place of l<layers> after the hidden width.
QEL fits and the SDDR fits whose device is a GPU take one entry of --gpus each; all other fits run on the CPU, --cpus
at a time. The DFL fits of kp and bm need a Gurobi license.

example:
  python -m scripts.run_reported qel --data DATA --out OUT --gpus 0 0 0 1 1 1
"""

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data import PROBLEMS
from data.split_inputs import write_test, write_train_valid
from src.reported import DFL_METHODS, HIDDEN, PAIRS, QEL_LAYERS, SDDR_EPOCHS, SDDR_STRUCTURES, sddr_settings

ROOT = Path(__file__).resolve().parents[1]


def fits(family, problems):
    """(fit name, arguments of the training command, "gpu" or "cpu") of every reported fit of the family."""
    for problem in problems:
        for hidden in HIDDEN:
            if family == "qel":
                cells = [(f"l{layers}-h{hidden}", {"layers": layers}, "gpu") for layers in QEL_LAYERS]
            elif family == "sddr":
                cells = [
                    (
                        f"h{hidden}-{s}-e{epochs}",
                        {"structure": s, "epochs": epochs},
                        sddr_settings(problem, hidden, s)["device"].replace("cuda", "gpu"),
                    )
                    for s in SDDR_STRUCTURES
                    for epochs in SDDR_EPOCHS
                ]
            else:
                cells = [(f"h{hidden}-{method}", {"method": method}, "cpu") for method in DFL_METHODS]
            for cell, argument, device in cells:
                for data_seed, model_seed in PAIRS:
                    seeds = {"data_seed": data_seed, "model_seed": model_seed}
                    name = f"{problem}-{cell}-d{data_seed}-m{model_seed}"
                    yield name, {"problem": problem, "hidden": hidden} | argument | seeds, device


def check_machine(family, jobs, a):
    """Stop before the queue starts if the datasets, a GPU index or the Gurobi license is missing."""
    for problem in a.problems:
        for data_seed, _ in PAIRS:
            if not (a.data / problem / str(data_seed) / "y.npy").is_file():
                raise SystemExit(f"{a.data / problem / str(data_seed)} is missing; run python -m data.regenerate first")
    if a.cpus < 1 or min(a.gpus) < 0:
        raise SystemExit("--cpus needs a positive number and --gpus indices from 0")
    if any(device == "gpu" for _, _, device in jobs):
        import torch  # imported here: a run without GPU fits does not need it

        if max(a.gpus) >= torch.cuda.device_count():
            raise SystemExit(
                f"--gpus {max(a.gpus)}: {torch.cuda.device_count()} CUDA GPU(s) are visible, numbered from 0"
            )
    if family == "dfl" and any(arguments["problem"] in ("kp", "bm") for _, arguments, _ in jobs):
        # in a process of its own: Gurobi would change how this process receives Ctrl-C
        check = [sys.executable, "-c", "import gurobipy; gurobipy.Model().dispose()"]
        result = subprocess.run(check, capture_output=True, text=True)
        if result.returncode:
            reason = result.stderr.strip().splitlines()[-1]
            raise SystemExit(f"the DFL fits of kp and bm need a Gurobi license: {reason}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_reported", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("family", choices=("qel", "sddr", "dfl"))
    parser.add_argument("--data", type=Path, required=True, help="datasets written by data.regenerate")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS, help="default: all three")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="GPU indices, one fit per entry; repeat an index (default: 0). "
        "The runner sets CUDA_VISIBLE_DEVICES of every fit itself",
    )
    parser.add_argument("--cpus", type=int, default=4, help="number of CPU fits that run at the same time (default: 4)")
    a = parser.parse_args()
    a.problems = list(dict.fromkeys(a.problems))
    out, inputs = a.out.resolve() / a.family, a.out.resolve() / a.family / "inputs"
    jobs = [job for job in fits(a.family, a.problems) if not (out / "fits" / job[0] / "test.json").exists()]
    if jobs:
        check_machine(a.family, jobs, a)
    print(f"{a.family}: {len(jobs)} fits to run, logs in {out / 'logs'}", flush=True)
    if not jobs:
        return
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for problem in a.problems:
        for data_seed, _ in PAIRS:
            write_train_valid(problem, data_seed, a.data, inputs)

    gpus = queue.Queue()
    for gpu in a.gpus:
        gpus.put(gpu)
    test_written, lock, stop, running = set(), threading.Lock(), threading.Event(), set()

    def call(module, log, visible, **arguments):
        options = [str(x) for k, v in arguments.items() for x in ("--" + k.replace("_", "-"), v)]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=visible)
        command = [sys.executable, "-m", module, *options]
        with open(log, "a") as f:
            f.write(" ".join(command) + "\n")
            f.flush()
            with lock:  # an interrupt either comes before the start or finds the process in `running`
                if stop.is_set():
                    return None
                # a session of its own: Ctrl-C reaches the runner only, which then ends the fits itself
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=f, start_new_session=True)
                running.add(process)
            try:
                return process.wait()
            finally:
                with lock:
                    running.discard(process)

    def run(job):
        name, arguments, device = job
        fit, log, started = out / "fits" / name, out / "logs" / f"{name}.log", time.time()
        gpu = gpus.get() if device == "gpu" else None
        try:
            if stop.is_set():
                return name, "not started"
            visible = "" if gpu is None else str(gpu)
            code, tested = 0, False
            if not (fit / "selection.json").exists():
                code = call(f"scripts.{a.family}_train", log, visible, **arguments, inputs=inputs, fit=fit)
            if code == 0 and not stop.is_set():
                dataset = arguments["problem"], arguments["data_seed"]
                with lock:  # the TEST part of a dataset is written once, before the first TEST that reads it
                    if dataset not in test_written:
                        write_test(*dataset, a.data, inputs)
                        test_written.add(dataset)
                code, tested = call(f"scripts.{a.family}_test", log, visible, fit=fit, inputs=inputs), True
            if tested and code == 0:
                result = "ok"
            else:
                result = "interrupted" if stop.is_set() else f"FAILED (exit {code}), see {log}"
        except BaseException as error:  # also SystemExit of write_test; the other fits still report their result
            result = f"FAILED ({error}), see {log}"
        finally:
            if gpu is not None:
                gpus.put(gpu)
        print(f"{name} {result} {time.time() - started:.0f}s", flush=True)
        return name, result

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with ThreadPoolExecutor(len(a.gpus)) as on_gpu, ThreadPoolExecutor(a.cpus) as on_cpu:
        futures = []
        try:
            for job in jobs:
                futures.append((on_gpu if job[2] == "gpu" else on_cpu).submit(run, job))
            results = [future.result() for future in futures]
        except KeyboardInterrupt:  # Ctrl-C or SIGTERM: start no further fit and end the running ones
            stop.set()
            with lock:
                for process in running:
                    process.terminate()
            results = [future.result() for future in futures]
            results += [(name, "not started") for name, _, _ in jobs[len(futures) :]]
    counts = {
        kind: sum(result.startswith(kind) for _, result in results) for kind in ("ok", "interrupted", "not started")
    }
    failed = [f"{name}: {result}" for name, result in results if result.startswith("FAILED")]
    print(
        f"{counts['ok']} fits finished, {counts['interrupted']} interrupted, {counts['not started']} not started",
        flush=True,
    )
    print(f"{len(failed)} failed", *failed, sep="\n", flush=True)
    sys.exit(counts["ok"] != len(results))


if __name__ == "__main__":
    main()
