# Reproducibility Package

## Overview

This repository contains the code and lightweight experimental outputs supporting the experimental validation presented in the accompanying manuscript. The repository is organised according to the four validation objectives (V1–V4), allowing each experiment to be reproduced independently.

Large intermediate datasets (e.g., raw embeddings, API outputs, or extensive experimental caches) are not included because they exceed GitHub storage limits. The repository contains the scripts required to regenerate them together with the lightweight results used in the manuscript.

## Repository structure

```
requirements.txt
rq1/
rq2/
rq3/
rq4/
```

Install the common dependencies with:

```bash
pip install -r requirements.txt
```

Each validation objective can then be executed independently.

---

# V1 – Controlled Evaluation of Expressive Approximation

**Directory:** `rq1/`

## Purpose

Validates the finite-complexity behaviour associated with expressive approximation.

## Main scripts

| Script | Purpose |
|--------|---------|
| `run_rq1.py` | Main expressive approximation experiment. |
| `run_rq1_validation.py` | Robustness validation using alternative discretisations. |

Outputs are generated under `results/` and `figures/`.

---

# V2 – Controlled Evaluation of Representational Limits

**Directory:** `rq2/`

## Purpose

Validates observational and representation-induced indistinguishability.

## Main script

| Script | Purpose |
|--------|---------|
| `run_rq2.py` | Executes the complete V2 experiment and generates all reported outputs. |

Outputs are generated under `results/` and `figures/`.

---

# V3 – Calibration under Controlled Distributional Perturbation

**Directory:** `rq3/`

## Purpose

Evaluates bootstrap calibration together with controlled location, scale and contamination perturbations.

## Main script

| Script | Purpose |
|--------|---------|
| `run_rq3.py` | Executes the complete calibration experiment. |

Outputs are generated under `results/` and `figures/`.

---

# V4 – Evaluation on Heterogeneous Stochastic Intelligent Systems

**Directory:** `rq4/`

## Purpose

Evaluates the framework across heterogeneous language models and a common observable representation.

## Main scripts

| Script | Purpose |
|--------|---------|
| `generate_rq4.py` | Generates the evaluation corpus. |
| `embed_rq4_incremental.py` | Obtains responses and embeddings. |
| `run_rq4_analysis.py` | Performs calibration and statistical analysis. |
| `analyse_rq4_convergence.py` | Convergence analysis. |
| `validate_rq4_sample_size.py` | Sample-size validation. |
| `validate_rq4_corpus.py` | Corpus validation. |

Supporting directories include `embeddings/`, `validation/`, `sample_size_analysis/`, `convergence_analysis/`, `rq4_analysis/`, `results/`, and `figures/`.

---

# Lightweight Results

The repository contains all figures, summary tables, and lightweight numerical outputs reported in the manuscript.

Large intermediate artefacts, including raw language-model generations, complete embedding collections, and large cached datasets, are intentionally omitted because they exceed GitHub storage limits. The scripts required to regenerate these artefacts are included.

---

# Reproducibility

Each validation objective is self-contained and can be executed independently.

Typical workflow:

1. Install the dependencies.
2. Execute the corresponding `run_*.py` script(s).
3. Inspect the generated outputs in the associated `results/` and `figures/` directories.

The repository structure mirrors the organisation of the experimental section of the manuscript, allowing reviewers to reproduce each validation objective independently.

