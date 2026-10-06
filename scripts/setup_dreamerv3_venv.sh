#!/bin/bash
# Build the DreamerV3 runtime (third_party/Offline_vs_Online_in_MBRL) as a Python 3.9 venv.
#
# Mirrors the repo's Dockerfile, restricted to what the DMC code path imports
# (no gym 0.19, mujoco-py, Atari, Metaworld). Versions are pinned to what the
# Docker image resolved to for Python 3.9: JAX 0.4.30 is the last release with
# Python 3.9 support, and TensorFlow 2.16 needs numpy < 2.
#
# Usage: scripts/setup_dreamerv3_venv.sh [python3.9 interpreter]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-/appl9/python/3.9.23/bin/python3.9}"
VENV="$ROOT/.venvs/dreamerv3"

uv venv --python "$PYTHON" "$VENV"
uv pip install --python "$VENV/bin/python" \
    "jax[cuda12]==0.4.30" \
    "numpy<2" \
    "scipy==1.12.0" \
    "optax==0.2.3" \
    "tensorflow-cpu==2.16.2" \
    "tensorflow-probability==0.24.0" \
    "tf-keras==2.16.0" \
    "dm-control==1.0.11" \
    "mujoco==2.3.3" \
    cloudpickle rich ruamel.yaml opencv-python-headless matplotlib pillow

"$VENV/bin/python" -c "import jax, tensorflow, tensorflow_probability, dm_control, optax; print('dreamerv3 venv OK, jax', jax.__version__)"
