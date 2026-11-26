# QEL paper experiments

This repository contains the code behind the results tables of the quantum end-to-end learning (QEL) paper. It is a
reproduction package: it regenerates the datasets, trains and tests the 600 reported fits, and nothing else.

Three families of methods are compared on budget allocation (BA), knapsack (KP) and bipartite matching (BM):

- **QEL** with circuit depths p = 4 and 6. A classical encoder predicts the coefficients of the objective from the
  features of an instance. The coefficients enter a QAOA-type circuit whose state stays in the span of the feasible
  decisions, and the encoder and the circuit angles are trained together on the expected energy of the true objective.
  The circuits are simulated exactly.
- **DFL baselines**: eight decision-focused learning methods (Blackbox [3], Identity [4], SPO [5], NCE [6], Point-LTR,
  Pair-LTR and List-LTR [7], and LODL [8]), run by the training loop of OpenPTO, the library of the
  PredictiveCO-Benchmark repository [1].
- **SDDR baselines**: the score-based discrete decision rule [2] with independent and with shared score networks, each
  trained with epoch budgets of 200, 400, 600, 800 and 1000.

QEL and the DFL baselines use one tanh MLP with one hidden layer of width 8 or 12, applied to every item of an instance
with shared weights. The SDDR score networks have the same hidden width but read the features of all items at once.

## Requirements

- Linux on x86-64, and [uv](https://docs.astral.sh/uv/) to install the pinned environment (Python 3.12, PyTorch 2.11
  with CUDA 12.8, JAX 0.11 with CUDA 12).
- An NVIDIA GPU for QEL and for the SDDR fits with independent scores on BM. A QEL fit uses about 0.5 GB of GPU
  memory on BA and KP and 2 GB on BM.
- gcc or clang on `PATH` for QEL on BM, which compiles one GPU kernel with `torch.compile`.
- A Gurobi license for the DFL baselines on KP and BM (`gurobipy` 12.0.3). Academic licenses are free; put the license
  where Gurobi looks for it (`~/gurobi.lic` or `GRB_LICENSE_FILE`). The restricted license inside the `gurobipy` wheel
  has expired and does not work. Gurobi solves the knapsack decisions inside OpenPTO and the exact matchings that
  evaluate the DFL baselines on BM on VALID and TEST (default MIP gap 1e-4). QEL, SDDR, the DFL baselines on BA and data
  regeneration do not use Gurobi.
- Disk: about 8 GB for the environment, 0.4 GB for the raw files, 0.75 GB for the datasets and 8 GB for the outputs.

On two Intel Xeon Gold 5317 processors (48 logical cores) and four NVIDIA GeForce RTX 4090 cards (24 GB), the three
commands under [Running](#running) take about 6 h 35 min in total.

## Setup

```bash
git clone https://github.com/jaehwan-2ee/qel-code.git && cd qel-code
uv sync --frozen
source .venv/bin/activate
```

Run every command from the root of the repository.

Every command that trains, tests or regenerates data imports `src`, which sets the runtime environment: one BLAS and
OpenMP thread, TF32 off, a deterministic cuBLAS workspace, and deterministic XLA GPU operations without autotuning.
The results tables were computed on Intel Xeon Gold 5317 processors and NVIDIA GeForce RTX 4090 cards
(driver 580.178.04) under Ubuntu 24.04. On a fixed machine the results are deterministic.

## Data

No raw or generated dataset is distributed. The datasets are regenerated from three public sources. Download the raw
files into a directory `RAW`:

```bash
mkdir -p RAW/ba RAW/kp RAW/bm
# BA, 400 MB (a Git LFS file of github.com/sanketkshah/LODLs)
curl -L -o RAW/ba/budget_allocation_data.pkl \
    https://media.githubusercontent.com/media/sanketkshah/LODLs/main/data/budget_allocation_data.pkl
# KP
curl -L -o RAW/kp/prices2013.dat \
    https://raw.githubusercontent.com/JayMan91/aaai_predit_then_optimize/master/prices2013.dat
# BM: the certificate chain of this host is incomplete, hence -k; the digests below are checked by data.regenerate
curl -kL -o cora.tgz https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
tar -xOzf cora.tgz cora/cora.cites > RAW/bm/cora.cites && tar -xOzf cora.tgz cora/cora.content > RAW/bm/cora.content
```

| Problem | File under RAW | sha256 | Source and terms |
|---|---|---|---|
| BA | `ba/budget_allocation_data.pkl` | `11a0ac25f526c11f9fee9808c78f77055e036022540a5dd433a90e10393c786f` | Yahoo! Webscope advertising data as distributed with the LODL code [8]. The OpenPTO paper [1] describes it as available to academics for non-commercial use under Yahoo's data sharing agreement. |
| KP | `kp/prices2013.dat` | `c8086b6b026c1edce01bea265392e42ce46245e9f7f5e88d871232e8c6176faf` | Irish electricity prices of the SEMO market [9], from the code repository of [10] (MIT License). That repository states no terms for the price data. |
| BM | `bm/cora.cites` | `ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e` | Cora citation network [11], LINQS release. The archive contains no license statement. `cora.zip` on the [LINQS datasets page](https://linqs.org/datasets/) holds the same two files. |
| | `bm/cora.content` | `b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee` | |

Then regenerate the 15 datasets (three problems, five data seeds):

```bash
python -m data.regenerate --raw RAW --data DATA
```

The command writes `DATA/<problem>/<data seed>/` with `X.npy`, `y.npy`, `weights.npy` (KP only) and `meta.json`, and
prints one line per dataset, for example `ba 2025 identical`. It compares each of the 35 array files with its pinned
sha256 (`configs/data_sha256.json`) and ends with an error if one differs. A
dataset that already exists is compared and not generated again. `--problems` and `--seeds` select a part.

The generators follow the data construction of OpenPTO (see the docstrings in `data/`). The KP TEST instances are the
same 200 days for every data seed; the data seed changes the TRAIN and VALID days only. For BA and BM the data seed
changes all three parts.

The only file in this repository that is derived from data is `data/bm/precomputed/cora_cites_metis.txt.part.135`, a
partition of the Cora citation graph into 135 parts with one part id per line. It was computed once on macOS (arm64)
with METIS 5.1.0 [12] through PyMetis 2025.2.2 (`nparts=135`, seed 2026). The graph was read from `cora.cites` with
NetworkX 3.6, its nodes numbered in order of first appearance, and the adjacency list of every node sorted; line i is
the part of node i.

Training does not read the dataset directories. `data.split_inputs` cuts every dataset into two parts, and all three
families read those: the TRAIN and VALID rows (with the KP weights) in `INPUTS/<problem>/<data seed>/`, and the TEST
rows (with the TEST optimum of BA and KP) in `INPUTS/<problem>/<data seed>/test/`. Both parts are checked against
`configs/split_inputs_sha256.json` when they are written and whenever they are read. The runner below writes them.

## Running

```bash
python -m scripts.run_reported qel  --data DATA --out OUT --gpus 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
python -m scripts.run_reported sddr --data DATA --out OUT --gpus 0 1 2 3 --cpus 16
python -m scripts.run_reported dfl  --data DATA --out OUT --cpus 44
```

The three commands train and test the 60 QEL fits, the 300 SDDR fits and the 240 DFL fits.

- `--gpus` lists GPU indices. Every entry runs one fit at a time, so an index that appears
  six times runs six fits on that GPU. Six QEL fits fit on a 24 GB card. The default is `--gpus 0`.
- `--cpus` is the number of CPU fits that run at the same time (default 4). Every CPU fit uses one thread.
- `--problems ba kp bm` restricts a run to some problems and needs only their datasets.
- Before it starts, the runner checks that the GPU indices exist and, for the DFL fits of KP and BM, that Gurobi has a
  usable license.
- Each fit logs to `OUT/<family>/logs/<fit>.log`. The runner prints one line per fit and, at the end, the fits that
  failed with the path of their log.
- The same command resumes a run: a fit that has its `test.json` is skipped, and a fit that was interrupted during
  training is trained again from the start. Ctrl-C ends the running fits and starts no further one.
- The three commands may share `OUT`. Run one runner per family and output directory at a time.

### One fit

The runner calls one training command and one TEST command per fit. For KP, p = 4, width 8 and the pair (2025, 1020):

```bash
python -m data.split_inputs train-valid --problem kp --data-seed 2025 --data DATA --inputs INPUTS
python -m scripts.qel_train --problem kp --layers 4 --hidden 8 --data-seed 2025 --model-seed 1020 \
    --inputs INPUTS --fit OUT/qel/fits/kp-l4-h8-d2025-m1020
python -m data.split_inputs test --problem kp --data-seed 2025 --data DATA --inputs INPUTS
python -m scripts.qel_test --fit OUT/qel/fits/kp-l4-h8-d2025-m1020 --inputs INPUTS
```

SDDR and the DFL baselines have the same two commands: `scripts.sddr_train` with `--structure {independent,shared}`
and `--epochs {200,400,600,800,1000}`,
`scripts.dfl_train` with `--method {blackbox,identity,spo,nce,point-ltr,pair-ltr,list-ltr,lodl}`, and
`scripts.sddr_test` and `scripts.dfl_test` with the arguments of `scripts.qel_test`. `--layers` is the circuit depth p.

## Outputs

`OUT/<family>/fits/<fit>/` holds these files:

| File | Written by | Content |
|---|---|---|
| `selection.json` | training | the fit, its settings, the selected epoch and the sha256 of the checkpoint |
| `epochs.jsonl` | training | the TRAIN loss and the VALID metric of every epoch (QEL: every joint epoch) |
| `selected.npy` (QEL), `selected.pt` | training | the reported checkpoint |
| `test.json` | TEST | `test_RR`, the mean TEST relative regret as a fraction; `test_count`, `infeasible_count`, `zero_optimum_count`; `test_RR_per_instance` |
| `test_arrays.npz` (SDDR, DFL) | TEST | scores or predictions, decisions and relative regret of every TEST instance |

`<fit>` is `<problem>-l<p>-h<width>-d<data seed>-m<model seed>` for QEL,
`<problem>-h<width>-<structure>-e<epochs>-d...-m...` for SDDR and `<problem>-h<width>-<method>-d...-m...` for the DFL baselines. QEL also writes `start.npy`,
`joint_start.npy` and `epoch_params.npy` (the parameters before training, after the warm-up stages and after every
joint epoch). `OUT/<family>/inputs/` holds the split inputs and `OUT/<family>/logs/` the logs.

## Repository layout

```
configs/       settings of QEL and SDDR, sha256 of the datasets and of the split inputs
data/          dataset generators, data.regenerate, data.split_inputs, the METIS partition of Cora
src/           QEL: protocol, encoder, the three circuits, SPSA, decoding; early stopping and relative regret
baselines/     SDDR (sddr*.py) and the DFL baselines on OpenPTO (dfl*.py)
third_party/   the vendored OpenPTO files (predictiveco/)
scripts/       <family>_train, <family>_test and run_reported
```

`PROVENANCE.json` records the sources of the vendored code and of the data, and `THIRD_PARTY_NOTICES.md` the code of
other projects that this repository contains or follows.

## References

1. H. Geng, H. Ruan, R. Wang, Y. Li, Y. Wang, L. Chen and J. Yan. Benchmarking PtO and PnO Methods in the Predictive
   Combinatorial Optimization Regime. Advances in Neural Information Processing Systems 37, 2024. (OpenPTO)
2. Y. Zhang, J. Liu and X. Zhao. Score-Based Discrete Decision Rules for Contextual Constrained Stochastic Programming.
   Operations Research Letters 68:107486, 2026. (SDDR)
3. M. Vlastelica Pogančić, A. Paulus, V. Musil, G. Martius and M. Rolínek. Differentiation of Blackbox Combinatorial
   Solvers. ICLR 2020. (Blackbox)
4. S. S. Sahoo, A. Paulus, M. Vlastelica, V. Musil, V. Kuleshov and G. Martius. Backpropagation through Combinatorial
   Algorithms: Identity with Projection Works. ICLR 2023. (Identity)
5. A. N. Elmachtoub and P. Grigas. Smart "Predict, then Optimize". Management Science 68(1):9-26, 2022. (SPO)
6. M. Mulamba, J. Mandi, M. Diligenti, M. Lombardi, V. Bucarey and T. Guns. Contrastive Losses and Solution Caching
   for Predict-and-Optimize. IJCAI 2021. (NCE)
7. J. Mandi, V. Bucarey, M. Mulamba and T. Guns. Decision-Focused Learning: Through the Lens of Learning to Rank.
   ICML 2022. (Point-LTR, Pair-LTR, List-LTR)
8. S. Shah, K. Wang, B. Wilder, A. Perrault and M. Tambe. Decision-Focused Learning without Decision-Making: Learning
   Locally Optimized Decision Losses. Advances in Neural Information Processing Systems 35, 2022. (LODL, BA data file)
9. G. Ifrim, B. O'Sullivan and H. Simonis. Properties of Energy-Price Forecasts for Scheduling. CP 2012. (KP data)
10. J. Mandi, E. Demirović, P. J. Stuckey and T. Guns. Smart Predict-and-Optimize for Hard Combinatorial Optimization
    Problems. AAAI 2020. (the KP problem)
11. P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher and T. Eliassi-Rad. Collective Classification in Network
    Data. AI Magazine 29(3):93-106, 2008. (Cora)
12. G. Karypis and V. Kumar. A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs. SIAM Journal
    on Scientific Computing 20(1):359-392, 1998. (METIS)

## License

The code of this repository is under the MIT License (`LICENSE`). The vendored OpenPTO files keep their own MIT
license (`third_party/predictiveco/LICENSE`); `THIRD_PARTY_NOTICES.md` names the projects from which parts of that code
originate. No raw or generated dataset is distributed, and the raw sources keep their own terms (see [Data](#data)).
