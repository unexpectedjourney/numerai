import datetime

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load

from src.metrics import (
    numerai_score,
    neutralized_numerai_score,
    autocorr_penalty,
    smart_sharpe,
    numerai_sharpe,
    adj_sharpe,
    numerai_score_and_sharpe,
)
from src.model import XGBoostModel, full_models
from src.preprocessing import (
    preprocessing,
    clear_era_records,
)
from src.split import TimeSeriesSplitGroups


class Trainer:
    def __init__(self, train_file, test_file, submissions_path, model_name,
                 model_params, save_path=None, plot_eras=False):
        # self.train_df = pd.read_csv(train_file, nrows=10000)
        # self.test_df = pd.read_csv(test_file, nrows=10000)
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)

        self.submission_df = self.test_df.copy()[["id"]]
        preproc_funcs = [
            clear_era_records,
        ]

        # self.kfold = TimeSeriesSplit(n_splits=5)
        self.kfold = TimeSeriesSplitGroups(n_splits=5)
        self.train_df, self.test_df = preprocessing(
            self.train_df,
            self.test_df,
            preproc_funcs,
        )

        today = datetime.date.today()
        date_str = '_' + str(today.day) + '_' + str(today.month) + '_' + str(today.year)
        self.save_path = str(save_path) + '/' + model_name + date_str + '.pickle'
        self.submissions_path = submissions_path
        self.model_name = model_name
        self.model_params = model_params

        self.model = self.get_model(model_name, model_params)
        self.era_metrics = [
            autocorr_penalty,
            smart_sharpe,
            numerai_sharpe,
            adj_sharpe
        ]
        self.plot_eras = plot_eras

    def load_model(self, path):
        self.model = load(path)

    @staticmethod
    def get_model(model_name, model_params):
        assert model_name in full_models, "Specify model which code contains"
        return full_models.get(model_name)(model_name, model_params)

    def train(self):
        self.model.train(
            self.train_df,
            self.kfold,
            # neutralized_numerai_score,
            numerai_score_and_sharpe,
            era_metrics=self.era_metrics,
            plot_eras=self.plot_eras,
        )

    def find_hyperparameters(self):
        self.model.find_hyperparameters(
            self.train_df,
            self.kfold,
            # neutralized_numerai_score,
            numerai_score_and_sharpe,
            target="target",
        )

    def evaluate(self):
        predictions = self.model.predict_and_score(
            self.train_df,
            self.kfold,
            self.test_df,
            "target",
            neutralized_numerai_score,
            era_metrics=self.era_metrics,
            plot_eras=self.plot_eras,
        )
        self.submission_df["prediction"] = predictions
        self.submission_df[["id", "prediction"]].to_csv(
            self.submissions_path /
            f"{self.model_name}-{int(datetime.datetime.now().timestamp())}.csv",
            index=False)

    def evaluate_for_submition(self):
        predictions = self.model.predict(
            self.train_df,
            self.kfold,
            self.test_df,
            "target",
            # neutralized_numerai_score,
            numerai_score_and_sharpe,
        )
        self.submission_df["prediction"] = predictions
        self.submission_df[["id", "prediction"]].to_csv(
            self.submissions_path /
            f"{self.model_name}-{int(datetime.datetime.now().timestamp())}.csv",
            index=False)

    def save_model(self):
        dump(self.model, self.save_path)