#!/bin/bash
#SBATCH --job-name=rag-smoke
#SBATCH --output=rag-smoke-%j.out
#SBATCH --error=rag-smoke-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail
source .venv/bin/activate
python scripts/run_experiment.py --config configs/smoke_test.yaml
