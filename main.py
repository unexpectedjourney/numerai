from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.metrics import numerai_score
from src.preprocessing import preprocessing, drop_columns, clear_era_records, \
    get_validation_data
from src.train import cross_validate
from src.utils import set_seed

random_state = 42


def main():
    set_seed(random_state)
    dir_path = Path("./data")
    train_file = dir_path / "numerai_training_data.csv"
    test_file = dir_path / "numerai_tournament_data.csv"
    submissions_path = Path("./submissions")

    train_df = pd.read_csv(train_file, nrows=1000)
    test_df = pd.read_csv(test_file, nrows=1000)

    preproc_funcs = [
        get_validation_data,
        drop_columns,
        clear_era_records,
    ]
    viz_losses = []

    kfold = TimeSeriesSplit(n_splits=5)
    train_df, test_df = preprocessing(train_df, test_df, preproc_funcs)
    base_model = XGBClassifier()
    val_scores, test_preds = cross_validate(
        base_model,
        train_df.copy(),
        kfold,
        numerai_score,
        test_df=test_df.copy(),
        viz_losses=viz_losses
    )


if __name__ == '__main__':
    main()
