from hyperopt import Trials, tpe, fmin
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import numpy as np
from hyperopt import hp

from src.train import cross_validate

N_ITERS = 20
models = {
    "xgboost": XGBClassifier,
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "linear_svc": LinearSVC,
    "gaussian_nb": GaussianNB,
    "ridge": RidgeClassifier,
}


class BaseModel:
    def __init__(self, model_name):
        self.model = None
        self.model_class = self.get_model(model_name)
        self.apply_hyper_params()
        self.params = self.get_params()
        self.model_list = []

    @staticmethod
    def get_model(model_name):
        return models.get(model_name)

    @staticmethod
    def get_params():
        return {}

    def apply_hyper_params(self, params=None):
        if params is not None and isinstance(params, dict):
            self.model = self.model_class(**params)
        else:
            self.model = self.model_class()

    def train(self, train_df, kfolds, metric):
        val_scores, self.model_list = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            return_models=True
        )
        return self

    def predict(self, test_df, target):
        assert len(self.model_list), "Train first"

        if target in test_df.columns:
            test_df = test_df.drop(target, axis=1)
        x_data = test_df.values

        predictions = np.column_stack(
            [
                model.predict(x_data) for model in self.model_list
            ]
        ).mean(axis=1)
        return predictions

    def predict_and_score(self, train_df, kfolds, test_df, target, metric,
                          additional_metrics=None):
        if additional_metrics is None:
            additional_metrics = []

        assert target in test_df.columns, "Please specify target in test_df"

        y_data = test_df[target].values

        val_scores, predictions = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            test_df=test_df,
        )
        predictions = stats.mode(np.array(predictions))[0][0]

        print(f"{metric.__name__}, {metric(y_data, predictions)}")
        for additional_metric in additional_metrics:
            print(
                f"{additional_metric.__name__}, {additional_metric(y_data, predictions)}")

        return predictions


class BoostingMixing:
    def find_hyperparameters(self, train_df, kfolds, metric):
        def try_hyperparameters(params):
            model = self.model_class(**params)
            return cross_validate(
                model,
                train_df.copy(),
                kfolds,
                metric,
                test_df=None,
            )

        # todo specify whether we need fmin or not
        result = fmin(
            fn=try_hyperparameters,
            space=self.params,
            trials=Trials(),
            algo=tpe.suggest,
            max_evals=N_ITERS,
            return_argmin=False,
        )
        print(result)


class SklearnModelMixing:
    def find_hyperparameters(self, train_df, target, kfolds, metric):
        splits = kfolds.split()
        search = RandomizedSearchCV(
            self.model, self.params, cv=splits, scoring=metric,
            return_train_score=True, verbose=True, n_jobs=-1, n_iter=N_ITERS
        )

        search.fit(train_df.drop(target, axis=1), train_df[target])
        print(search.best_params_, search.best_score_)


class XGBoostModel(BaseModel, BoostingMixing):
    @staticmethod
    def get_params():
        return {
            'max_depth': hp.choice('max_depth', range(5, 30, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
        }


class LogRegModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_params():
        return {}


class RFModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_params():
        return {}


class LSVCModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_params():
        return {}


class GaussianNBModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_params():
        return {}


class RidgeNBModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_params():
        return {}


full_models = {
    "xgboost": XGBoostModel,
}
