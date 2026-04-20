# NL-to-SQL

A natural language to SQL translation system using sequence labelling, built with PyTorch and scikit-learn. The system converts natural language queries into SQL by predicting both the SQL template and the variable tags for each token.

## Models

Two approaches are implemented:

### 1. Linear Model
A two-stage pipeline using scikit-learn:
- **Template classifier:** Predicts the SQL template from the input using TF-IDF vectorization and LinearSVC.
- **Tag classifier:** Predicts per-word variable tags using Logistic Regression.

### 2. LSTM Model
A neural sequence labelling model built with PyTorch:
- Embedding layer followed by a single-layer LSTM.
- Two output heads trained jointly: one for per-word tag prediction (BIO-style) and one for SQL template prediction from the final hidden state.
- Trained with a combined cross-entropy loss, gradient clipping, and a step learning rate scheduler.

## Data

The dataset is derived from GeoQuery, a benchmark dataset for natural language to SQL parsing, originally introduced by Zelle & Mooney (1996) at the UT Austin ML Group. The version used in this project is a cleaned and refined adaptation where the original Prolog logical forms have been translated to SQL. The dataset is provided as pre-split train, dev, and test files and is excluded from this repository.

## Usage

```bash
python main.py
```

By default, `main.py` runs the LSTM model. Switch to the linear model by calling `main_linear()` instead of `main_LSTM()` in the `__main__` block.
