import argparse

from Model_Linear import linear_model
from ModelLSTM import main_lstm
from ModelBiLSTM import main_bilstm

MODELS = {
    "linear": linear_model,
    "lstm": main_lstm,
    "bilstm": main_bilstm,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Natural language to SQL translation.")
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="bilstm",
        help="Which model to run (default: bilstm).",
    )
    args = parser.parse_args()
    MODELS[args.model]()
