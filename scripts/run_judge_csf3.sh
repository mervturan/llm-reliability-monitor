#!/bin/bash --login

#SBATCH --job-name=rag-judge
#SBATCH --output=rag-judge-%j.out
#SBATCH --error=rag-judge-%j.err
#SBATCH -p gpuL
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-2
#SBATCH --mem=24G

set -euo pipefail

cd /net/scratch/j52068mt/msc-thesis
source venv/bin/activate

PREDICTIONS_FILE=$1
LIMIT=${2:-}

echo "Predictions file: $PREDICTIONS_FILE"

if [[ -n "$LIMIT" ]]; then
    echo "Judge limit: $LIMIT"

    python scripts/evaluate_with_judge.py \
        --predictions "$PREDICTIONS_FILE" \
        --limit "$LIMIT"
else
    python scripts/evaluate_with_judge.py \
        --predictions "$PREDICTIONS_FILE"
fi