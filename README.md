# NL-to-SQL
Sequence labelling model using a LSTM to translate natural language queries into SQL, built with PyTorch

## Data
The dataset is derived from GeoQuery, a benchmark dataset for natural language to SQL parsing, originally introduced by Zelle & Mooney (1996) at the UT Austin ML Group. The version used in this project is a cleaned and refined adaptation, where the original Prolog logical forms have been translated to SQL. The dataset was provided as pre-split train, dev, and test files.
