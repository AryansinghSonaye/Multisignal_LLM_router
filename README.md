# Multi-Signal LLM Difficulty Router

A machine learning project exploring how to classify the difficulty of incoming LLM requests into **EASY, MEDIUM, and HARD** tiers for cost-aware model routing.

## Problem

Long prompts do not necessarily contain difficult tasks. A simple coding request may include thousands of tokens of source code, logs, tests, and documentation, while a short mathematical question may require substantial reasoning.

This project explores how to distinguish task difficulty from context volume using multiple structural and semantic signals.

## Approach

The pipeline extracts eight rule-based structural features and combines them with TF-IDF text features.

The structural features capture:
- Context volume and cumulative prompt length
- Conversation depth and multi-turn interactions
- Error indicators and code presence
- Modality indicators
- Python AST density as a structural code signal

These features are combined with 1,000 TF-IDF unigram and bigram features. A standardized feature-fusion pipeline feeds a logistic regression classifier with L2 regularization and balanced class weights.

## Dataset and Results

The system was evaluated on a held-out set of 5,475 prompts from RouterBench.

| Metric | Reported result |
|---|---:|
| Accuracy | 84.0% |
| Macro F1-score | 0.84 |
| HARD-tier precision | 0.94 |
| HARD-tier recall | 0.76 |

These results are from the project's documented evaluation and have not been independently reproduced here.

## Limitations

- The target labels are derived from cost-efficiency scores and are proxies for task difficulty.
- The contribution of context-volume features still needs to be tested through ablation.
- Rule-based feature extraction can produce false positives.
- AST-based analysis currently focuses on Python code.

## Repository Contents

The repository includes the implementation notebook, project report, presentation, and supporting problem-analysis document.

## Future Work

- Ablate context-volume features to test their influence on predictions.
- Evaluate performance on a controlled short/long and easy/hard benchmark.
- Measure feature-extraction and inference latency.
- Improve structural signals and extend code analysis to additional languages.

## Technologies

Python, scikit-learn, TF-IDF, logistic regression, feature engineering, and Jupyter Notebook.
