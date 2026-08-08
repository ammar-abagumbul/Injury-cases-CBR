# Research Plan: Learning Structured Representations of Personal Injury Cases

## Objective

The objective of this project is **not** to build the best feature extraction pipeline.

The objective is to determine:

> **What structured representation of a personal injury case best supports similarity retrieval and downstream prediction while remaining interpretable to legal practitioners?**

The project proceeds in sequential stages. Each stage has clearly defined experiments. Once a stage is completed, the selected approach is frozen and becomes the input for the following stage.

---

# Stage 1 — Reliable Structured Feature Extraction

## Goal

Develop and evaluate a robust extraction pipeline capable of reliably extracting all required structured features defined by the Law Faculty.

The extraction output should conform to Pydantic schemas.

This stage does **not** investigate retrieval or prediction.

---

## Input

Raw judgment documents.

Existing human-extracted CSV.

20–25 manually curated gold-standard cases.

---

## Output

One JSON object per case.

Example

```python
Case(
    metadata=...,
    plaintiff=...,
    injuries=[...],
    losses=[...],
    awards=[...]
)
```

Refer to `./FEATURES.md` and `./REVISED_LOSS.md` for the list of features and losses to extract. 

---

## Additional Representation

Introduce one additional object:

```python
InjuryLossRelation
```

Example

```python
InjuryLossRelation(
    injury="Rotator cuff tear",
    loss="Loss of congenial employment",
    evidence=[paragraph 62]
)
```

The relation is extracted only when explicitly supported by the judgment.

No inference is performed.

---

## Experiments

### Experiment 1.1 — Model Comparison

Independent Variable

- GPT
- Claude
- Gemini
- Local model (qwen3, deepseek) using olama

NOTE: we shall use langchain to access all of these models. 

Controlled Variables

- Same prompt
- Same schema
- Same documents

Evaluation

- Precision
- Recall
- F1
- Runtime
- Token count

---

### Experiment 1.2 — Prompt Design

Independent Variable

- Zero-shot
- Few-shot

Controlled Variables

- Same model
- Same schema

Evaluation

Same metrics as above.

---

### Experiment 1.3 — Extraction Strategy

Independent Variable

- Single-pass extraction
- Section-by-section extraction
- Multi-agent extraction

Evaluation

- Extraction accuracy
- Relationship accuracy
- Runtime

---

### Experiment 1.4 — Schema Validation

Independent Variable

- Raw JSON
- Pydantic validation
- Pydantic + repair pass

Evaluation

- Validation failures
- Repair success rate
- Hallucination rate

---

### Experiment 1.5 — Injury-Loss Relation Extraction

Independent Variable

Prompt design.

Evaluation

Edge Precision

Edge Recall

Edge F1

---

## Deliverables

- Best extraction model
- Best prompt
- Best extraction strategy
- Final Pydantic schema
- Gold benchmark dataset

---

## Exit Criteria

Freeze the extraction pipeline.

No further extraction changes are permitted.

---

# Stage 2 — Case Representation

## Goal

Investigate how different structured representations affect downstream retrieval.

Extraction remains fixed.

Only representation changes.

---

## Representation A

Flat Feature Vector

```
Age

Gender

Occupation

ICD Injury Codes

Losses

Awards
```

---

## Representation B

Weighted Feature Vector

Same features.

Each feature receives a similarity weight.

Initial weights may be manually assigned.

Later experiments may optimise them.

---

## Representation C

Flat Features + Injury-Loss Relations

```
Features

+

Injury → Loss edges
```

Only Injury → Loss edges are included.

No unrestricted graph is constructed.

---

## Research Questions

Does adding Injury-Loss relationships improve retrieval?

How much additional complexity is justified?

---

## Deliverables

Three representations.

---

## Exit Criteria

Select the best representation.

Freeze it.

---

# Stage 3 — Similar Case Retrieval

## Goal

Evaluate different similarity algorithms using the selected representation.

Representation remains fixed.

Only retrieval changes.

---

## Retrieval Method 1

Exact Feature Matching

Similarity computed from feature overlap.

---

## Retrieval Method 2

Weighted Feature Similarity

Similarity

=
weighted feature overlap.

Weights obtained from Stage 2.

---

## Retrieval Method 3

Weighted Feature Similarity

+

Injury-Loss Relation Similarity

Cases sharing

```
Rotator cuff tear

↓

Loss of congenial employment
```

receive additional similarity score.

---

## Evaluation Dataset

Create approximately 25 query cases.

For each query:

Law experts rank retrieved cases.

Relevance labels

- Relevant
- Partially Relevant
- Not Relevant

---

## Evaluation Metrics

Recall@5

Recall@10

Precision@10

MRR

NDCG

Expert agreement

---

## Deliverables

Best similarity algorithm.

---

## Exit Criteria

Freeze retrieval algorithm.

---

# Stage 4 — Prediction

## Goal

Determine whether richer representations improve predictive performance.

Retrieval remains fixed.

Representation remains fixed.

Only predictive models change.

---

## Prediction Tasks

Task 1

Predict award amount.

Task 2

Predict presence of future income loss.

Task 3

Predict major loss categories.

---

## Models

Linear Regression

Random Forest

Gradient Boosting

Explainable Boosting Machine

XGBoost

---

## Experiments

Compare prediction using

Representation A

Representation B

Representation C

---

## Evaluation

Regression

- MAE
- RMSE
- R²

Classification

- Precision
- Recall
- F1
- ROC AUC

Interpretability

- SHAP
- Feature Importance

---

## Deliverables

Best predictive model.

Feature importance analysis.

---

# Project Timeline

```
Stage 1

Reliable Extraction

↓

Stage 2

Representation

↓

Stage 3

Retrieval

↓

Stage 4

Prediction
```

Each stage is completed before the next begins.

No stage modifies previous stages.

---

# Expected Contributions

1. A benchmark dataset for structured feature extraction in personal injury judgments.

2. A reproducible extraction pipeline based on Python, Pydantic, and LLMs.

3. A comparative evaluation of structured case representations.

4. An evaluation of feature-based similarity retrieval for legal precedents.

5. An empirical study of the impact of Injury-Loss relationships on retrieval and prediction.

---

# Implementation Stack

Language

- Python

Libraries

- Pydantic
- Instructor (optional)
- Openrouter
- Pandas
- Scikit-learn
- XGBoost
- SHAP

Storage

- PostgreSQL
- JSON

Evaluation

- Pytest
- Jupyter
- Weights & Biases (optional)

Visualization

- Matplotlib
- Plotly
