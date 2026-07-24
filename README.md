# llm-reliability-monitor
MSc thesis project at The University of Manchester exploring conformal prediction for uncertainty-aware LLM monitoring and reliable evaluation of LLM-generated outputs.


# Instructions to Run the RAG Monitoring Starter

A reproducible starting point for the MSc thesis experiments on uncertainty-aware monitoring and escalation in retrieval-augmented question answering.

## First run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The run saves its full configuration, environment, row-level predictions, thresholds, metrics, and a CSV summary under `outputs/<run_name>/`.

## What it currently implements

- SQuAD calibration and test samples
- dense retrieval with SentenceTransformers
- answer generation with Qwen2.5-0.5B-Instruct
- retrieval, answer-context similarity, and consistency signals
- exact match and token F1
- an interpretable empirical-risk threshold baseline
- accept, flag, and escalate decisions

The threshold baseline is deliberately simple. It gives you a functioning experimental pipeline before replacing it with CTD, split conformal prediction, or conformal risk control.

