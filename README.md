# Credit Card Fraud Detection

Machine learning experiments for detecting fraudulent credit card transactions using classical ML models, boosting methods, and fraud-specific evaluation metrics.

## Repository Category

**Original / Portfolio Project**

This repository is part of my machine learning portfolio. It focuses on practical fraud detection using tabular transaction data and compares several supervised learning methods.

## Overview

Credit card fraud detection is a highly imbalanced classification problem. Fraud cases are rare compared with legitimate transactions, so models must be evaluated carefully using metrics such as AUC, precision, recall, and fraud-specific business tradeoffs.

This project includes a simple baseline version and a more complete modeling workflow.

## Dataset

The dataset contains credit card transactions made in September 2013 by European cardholders. It includes 284,807 transactions, with 492 fraud cases. Fraudulent transactions represent about 0.172% of all transactions, making the dataset highly imbalanced.

The input variables are numerical and mostly anonymized using PCA transformation:

- `V1` to `V28`: PCA-transformed principal components
- `Time`: seconds elapsed between each transaction and the first transaction
- `Amount`: transaction amount
- `Class`: target variable, where `1` indicates fraud and `0` indicates a legitimate transaction

## Notebooks

### 1. Simple Version

Random Forest and Logistic Regression are applied as baseline models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alimohammadi44/Credit-Card-Fraud-Detection/blob/main/credit_card_fraud_detection_Simple_Case.ipynb)

### 2. Complete Version

A more complete modeling workflow compares multiple machine learning methods.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alimohammadi44/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection_Predictive_Models.ipynb)

## Model Results

| Model | Data Split / Validation Strategy | Validation AUC | Test AUC | Notes |
|---|---|---:|---:|---|
| Random Forest Classifier | Train / Test | — | 0.85 | Baseline ensemble model |
| AdaBoost Classifier | Train / Test | — | 0.83 | Lower performance than Random Forest |
| CatBoost Classifier | Train / Test, 500 iterations | — | 0.86 | Improved performance with boosting |
| XGBoost | Train / Validation / Test | 0.984 | 0.974 | Best overall performance; validation used for early stopping |
| LightGBM | Train / Validation / Test | 0.974 | 0.946 | Strong performance with split validation |
| LightGBM with Cross-Validation | Cross-validation + Test | — | 0.93 | Slightly lower but more robust estimate |

## Key Takeaways

- The dataset is strongly imbalanced, so AUC and recall/precision tradeoffs are more useful than accuracy alone.
- Tree-based boosting models such as XGBoost, LightGBM, and CatBoost performed strongly.
- XGBoost achieved the highest predictive performance in these experiments.
- Cross-validation produced a more conservative but robust performance estimate.

## Why XGBoost Works Well for Fraud Detection

Fraud detection datasets often contain tabular features, nonlinear interactions, noisy labels, and class imbalance. XGBoost is a strong fit because it:

- performs well on structured/tabular data
- captures feature interactions automatically
- includes regularization to reduce overfitting
- supports imbalance handling through parameters such as `scale_pos_weight`
- requires limited preprocessing compared with many neural models
- supports practical interpretability through feature importance and SHAP explanations

## Future Extensions

### Graph Neural Networks

GNNs can be useful when fraud is relational. For example, a graph can connect:

```text
customer ↔ card ↔ device ↔ IP ↔ merchant ↔ transaction
```

This can help detect fraud rings, collusion, mule networks, shared devices, and synthetic identities.

### Sequence Models

Sequence models such as RNNs or Transformers may be useful when fraud depends on transaction histories, spending behavior over time, velocity patterns, or sudden behavioral changes.

### LLM-Assisted Fraud Workflows

Large language models are usually not the best direct classifier for pure numeric fraud data, but they can help with:

- merchant description normalization
- investigator summaries
- customer support or dispute-note analysis
- feature enrichment from text
- explanation generation

## Recommended Next Steps

- Add precision-recall curves
- Add confusion matrices for different decision thresholds
- Add SHAP explainability plots
- Add cost-sensitive fraud evaluation
- Add a reproducible `requirements.txt`
- Add a clean `src/` folder for reusable model code

## Author

Ali Mohammadi — [@alimohammadi44](https://github.com/alimohammadi44)
