"""Quantum end-to-end learning (QEL).

Importing this package sets the runtime environment of every fit, so that `python -m ...` needs no shell setup: one
BLAS/OpenMP thread, TF32 off, a deterministic cuBLAS workspace, deterministic XLA GPU operations and no XLA autotuning.
"""

import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("JAX_PLATFORMS", "cuda")

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
_flags = os.environ.get("XLA_FLAGS", "").split()
for _flag in ("--xla_cpu_multi_thread_eigen=false", "--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0"):
    if _flag not in _flags:
        _flags.append(_flag)
os.environ["XLA_FLAGS"] = " ".join(_flags)
