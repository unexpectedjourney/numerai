from pathlib import Path

import fire
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.metrics import numerai_score
from src.preprocessing import (
    preprocessing, drop_columns, clear_era_records, get_validation_data)
from src.train import cross_validate
from src.trainer import Trainer
from src.utils import set_seed

random_state = 42


def main(
        train=False,
        tune_params=False,
        evaluate=False,
):
    set_seed(random_state)
    dir_path = Path("./data")
    train_file = dir_path / "numerai_training_data.csv"
    test_file = dir_path / "numerai_tournament_data.csv"
    submissions_path = Path("./submissions")

    trainer = Trainer(train_file, test_file, submissions_path)
    if train:
        trainer.train()
    elif tune_params:
        trainer.find_hyperparameters()
    elif evaluate:
        trainer.evaluate()
    else:
        print("Bye!")


if __name__ == '__main__':
    fire.Fire(main)
