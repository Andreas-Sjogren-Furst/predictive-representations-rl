#!/bin/bash
# Submit an LSF job that builds the ExORL RND train/val hdf5 files used by onestep-fb for one domain.
# Requires the raw buffer from: sh third_party/onestep-fb/data_gen_scripts/exorl_download.sh <domain> rnd
#
# Usage (from the project root):  scripts/lsf/generate_exorl_dataset.sh point_mass_maze [data_dir]
set -euo pipefail

DOMAIN="${1:?usage: $0 <domain> [data_dir]}"
DATA_DIR="${2:-$HOME/.exorl/data}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -z "${LSF_ENVDIR:-}" ]; then
    set +eu; source /lsf/conf/profile.lsf; set -eu  # the LSF profile reads unset variables and returns non-zero
fi
mkdir -p "$ROOT/logs"

# The job script is generated here because DTU's LSF does not forward the submitting shell's variables.
bsub <<EOF
#!/bin/bash
#BSUB -J exorl_${DOMAIN}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o $ROOT/logs/exorl_${DOMAIN}_%J.out
#BSUB -e $ROOT/logs/exorl_${DOMAIN}_%J.out
set -euo pipefail

cd "$ROOT/third_party/onestep-fb"
export PYTHONPATH=.
export MUJOCO_GL=egl

.venv/bin/python data_gen_scripts/generate_exorl_dataset.py \\
    --domain_name=$DOMAIN --num_workers=16 \\
    --save_path=$DATA_DIR/rnd-$DOMAIN.hdf5

.venv/bin/python data_gen_scripts/generate_exorl_dataset.py \\
    --domain_name=$DOMAIN --num_workers=16 \\
    --save_path=$DATA_DIR/rnd-$DOMAIN-val.hdf5 --skip_size=5_000_000 --dataset_size=100_000
EOF
