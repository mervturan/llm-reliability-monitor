#!/bin/bash --login
#SBATCH --job-name=rag-combined
#SBATCH --output=rag-combined-%j.out
#SBATCH --error=rag-combined-%j.err
#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-4
#SBATCH --mem=24G

set -euo pipefail

cd /net/scratch/j52068mt/msc-thesis
source venv/bin/activate

CONFIG_FILE=$1

echo "Running config: $CONFIG_FILE"

python scripts/run_experiment.py \
    --config "$CONFIG_FILE"