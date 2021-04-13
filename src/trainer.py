import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.metrics import numerai_score
from src.model import XGBoostModel
from src.preprocessing import get_preproc_functions, preprocessing


class Trainer:
    def __init__(self, train_file, test_file, submissions_path):
        self.train_df = pd.read_csv(train_file, nrows=1000)
        self.test_df = pd.read_csv(test_file, nrows=1000)
        self.preproc_funcs = get_preproc_functions()

        self.kfold = TimeSeriesSplit(n_splits=5)
        self.train_df, self.test_df = preprocessing(
            self.train_df,
            self.test_df,
            self.preproc_funcs)
        self.model = XGBoostModel("xgboost")

    def train(self):
        self.model.train(self.train_df, self.kfold, numerai_score)

    def find_hyperparameters(self):
        self.model.find_hyperparameters(self.train_df, self.kfold, numerai_score)

    def evaluate(self):
        self.model.predict_and_score(self.train_df, self.kfold, self.test_df, "target", numerai_score)
