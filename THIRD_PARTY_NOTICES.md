# Third-party notices

The code of this repository is under the MIT License in `LICENSE`. This file lists the code of other projects that the
repository contains or follows.

## OpenPTO (PredictiveCO-Benchmark)

`third_party/predictiveco/` holds 41 files of [PredictiveCO-Benchmark](https://github.com/Thinklab-SJTU/PredictiveCO-Benchmark)
at commit `f885426f609288a3fbd2064cd2bf4b4138164745`: 34 unchanged and 7 edited. `third_party/predictiveco/UPSTREAM.json`
lists every file and describes every edit. The project is under the MIT License, Copyright (c) 2023 OpenPTO; its license
file is `third_party/predictiveco/LICENSE`.

That license file does not name the projects from which parts of the OpenPTO code are taken. They are the following.

- **LODL** ([sanketkshah/LODLs](https://github.com/sanketkshah/LODLs), MIT License, Copyright (c) 2023 Sanket Shah).
  Parts of `openpto/method/Models/LODLs.py`, `openpto/method/Solvers/neural/submodular.py`,
  `openpto/method/Solvers/neural/BudgetallocSolver.py`, `openpto/method/Solvers/utils_solver.py`,
  `openpto/method/Solvers/cvxpy/cp_bmatching.py`, `openpto/method/Models/MSE.py`, `openpto/method/Predicts/dense.py`,
  `openpto/problems/BudgetAllocation.py`, `openpto/problems/BipartiteMatching.py` and `openpto/problems/PTOProblem.py`.
- **PredOpt-Benchmarks** ([PredOpt/predopt-benchmarks](https://github.com/PredOpt/predopt-benchmarks), MIT License,
  Copyright (c) 2021 PredOpt-Benchmarks). The preprocessing of the energy price table in `openpto/problems/Knapsack.py`.
- **PyEPO** ([khalil-research/PyEPO](https://github.com/khalil-research/PyEPO), MIT License, Copyright (c) 2021). The
  solver base classes in `openpto/method/Solvers/abcptoSolver.py` and `openpto/method/Solvers/grb/grbSolver.py`, and the
  loss classes in `openpto/method/Models/Blackbox.py`, `Identity.py`, `SPO.py`, `NCE.py` and `LTR.py`.

All three use the MIT License, whose text is:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
> persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
> Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The dataset generators in `data/` are written for this repository. They follow the data construction of the OpenPTO
problem classes named in their docstrings, and `data/kp_gen.py` contains the 48 item weights of
`openpto/problems/Knapsack.py`.

## SDDR

`baselines/sddr*.py` and `scripts/sddr_*.py` implement the score-based discrete decision rule of Zhang, Liu and Zhao
for the three problems of this repository. They are written for this repository and contain no source code of the
authors' repository [zhangyiy0223/SDDR](https://github.com/zhangyiy0223/SDDR), which has no license file. Two settings
are taken from its script `main_constr_basic.py` at commit `f111485702fc5d1e8f17ff5d4f554c45a4abf62c`: the margin 0.3
(`constr_eps`) and the inverse-temperature schedule 1 + 3 t / T (`gamma_schedule`). The tanh units, the
shared variant, the decision rules of the three problems and the penalties for KP and BM are adaptations made in this
repository; the authors' code covers another problem, shortest paths on a grid.

## Data

No raw or generated dataset is distributed. The only file derived from data is
`data/bm/precomputed/cora_cites_metis.txt.part.135`, a partition of the Cora citation graph that was computed for this
repository (see the Data section of `README.md`). The raw sources and their terms are listed there.
