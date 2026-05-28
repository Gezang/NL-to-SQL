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

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a model via the `--model` flag (choices: `linear`, `lstm`, `bilstm`):

```bash
python main.py --model bilstm
```

The default model is `bilstm`. Test-set predictions are written to the `results/` directory.

## Results

All models are evaluated on the dev set. The headline metric is **Complete SQL Accuracy** (the full predicted query, template plus filled-in variables, matches the reference exactly).

| Model  | Complete SQL Accuracy | Template Accuracy | Tag Accuracy |
| ------ | --------------------- | ----------------- | ------------ |
| Linear | 40.8%                 | 51.0%             | 97.4%        |
| LSTM   | 46.9%                 | 49.0%             | 99.2%        |
| BiLSTM | 51.0%                 | 53.1%             | 99.2%        |

The BiLSTM performs best on full SQL prediction (~51% vs ~46% for the LSTM and ~40% for the linear baseline), as expected: bidirectional context lets the model condition each token's tag on both the preceding and following words, which helps both template selection and variable tagging.

## Future Work

- **Transformer-based model.** Replace the recurrent encoder with a transformer (e.g. a fine-tuned encoder-decoder such as T5, or an encoder like BERT for the tagging/template heads). Self-attention captures long-range dependencies more directly than an LSTM and is the current standard for text-to-SQL.
- **Original GeoQuery dataset.** Train and evaluate on the original GeoQuery dataset rather than the cleaned SQL adaptation used here, to enable comparison against published benchmarks.
- **Subword tokenization.** Implement a WordPiece/BPE tokenizer in place of whitespace tokenization to reduce out-of-vocabulary tokens, and measure whether it improves performance.
