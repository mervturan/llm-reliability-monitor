# llm-reliability-monitor

MSc thesis project at The University of Manchester exploring uncertainty-aware monitoring and delegation for LLM-based retrieval-augmented question answering.

The project evaluates calibrated monitoring and cascade strategies for deciding when answers from a smaller language model can be accepted and when queries should be escalated to a larger model.


## Setup

Create and activate a virtual environment, install the dependencies, and run a smoke test:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Each experiment saves its configuration, environment information, row-level predictions, calibrated thresholds, evaluation metrics, and summary outputs under:

```text
outputs/<run_name>/
```


## Running Experiments

Experiments are controlled using YAML configuration files.

A local experiment can be run with:

```bash
python scripts/run_experiment.py --config <path-to-config.yaml>
```

For example:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```


## Running Experiments on CSF3

Experiments can be submitted to the University of Manchester CSF3 cluster using the provided SLURM script.

For example:

```bash
sbatch scripts/run_csf3.sh configs/baseline/combined_100.yaml
```

Monitor submitted jobs with:

```bash
squeue -u $USER
```

Monitor the output of a running job with:

```bash
tail -f rag-<JOBID>.out
```

Experiment outputs are written to the `outputs/` directory.


## Experimental Setup

The current experiments use:

- **Dataset:** SQuAD
- **Retrieval:** SentenceTransformers-based dense retrieval
- **Small model:** Qwen2.5-0.5B-Instruct
- **Large model:** Qwen2.5-3B-Instruct
- **Pipeline:** retrieval-augmented question answering
- **Delegation:** calibrated small-to-large model cascading

The smaller model generates an initial answer using retrieved evidence. Monitoring signals are then used to determine whether the answer should be accepted, flagged, or escalated to the larger model.

Calibration data is used to determine decision thresholds before evaluation on the held-out test set.


## Monitoring Signals

The framework supports reliability and uncertainty signals including:

- retrieval quality
- answer-context similarity
- faithfulness / entailment
- self-consistency
- semantic similarity
- model self-confidence

These signals can be combined and used by different monitoring or delegation strategies.


## Delegation Decisions

The monitoring framework supports three main outcomes:

- **Accept** — use the smaller model's answer
- **Flag** — identify the answer as uncertain or potentially unreliable
- **Escalate** — delegate the query to the larger model

This allows different monitoring and cascade strategies to be compared while keeping the underlying dataset, retrieval pipeline, model pair, and evaluation procedure fixed.


## Evaluation

The evaluation framework includes:

- Exact Match (EM)
- token F1
- semantic similarity
- LLM-as-a-Judge
- faithfulness
- answer relevance
- selective risk / selective accuracy
- escalation rate
- latency
- estimated inference cost
- calibration behaviour

Together, these metrics are used to evaluate both answer quality and the behaviour of the monitoring and delegation policies.


## LLM-as-a-Judge Evaluation

LLM-as-a-Judge evaluation is performed separately from the main experiment run using:

```text
scripts/evaluate_with_judge.py
```

The evaluator reads saved experiment predictions and adds judge-based quality assessments to the experiment outputs.

This separation allows the main generation and monitoring experiments to be completed independently from the more expensive judge evaluation stage.


## Outputs

Each experiment creates an output directory under:

```text
outputs/<run_name>/
```

Depending on the experiment configuration, this directory contains artifacts such as:

- saved experiment configuration
- environment and reproducibility information
- calibration thresholds
- row-level predictions
- monitoring signals and decisions
- aggregate evaluation metrics
- judge evaluation results
- CSV summaries

These outputs are used for comparison and analysis across experimental configurations.


## Current Scope

The main experimental focus is on comparing uncertainty-aware monitoring and delegation strategies within a common RAG question-answering pipeline.

The experimental framework keeps the underlying components fixed where possible so that differences in performance can be attributed to the delegation policy rather than changes to the dataset, retrieval system, models, prompts, or evaluation procedure.

SQuAD is used as the primary question-answering dataset, with additional experiments and extensions evaluated separately where applicable.