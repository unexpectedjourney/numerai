import numpy as np
from hyperopt import Trials, tpe, fmin
from hyperopt import hp
from scipy.stats import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor

from src.train import cross_validate
from src.utils import prepare_wrap_metric

N_ITERS = 40
models = {
    "xgboost": XGBRegressor,
    "lr": LinearRegression,
    "e_net": ElasticNet,
    "lasso": Lasso,
    "ridge": Ridge,
    "l_svr": LinearSVR,
    "svr": SVR
}


class BaseModel:
    def __init__(self, model_name, model_params):
        self.model = None
        self.model_class = self.get_model(model_name)
        self.params = model_params
        self.model = self.apply_hyper_params(self.model_class, self.params)
        self.tune_params = self.get_tune_params()
        self.model_list = []

    @staticmethod
    def get_model(model_name):
        return models.get(model_name)

    @staticmethod
    def apply_hyper_params(model_class, params=None):
        if params is not None and isinstance(params, dict):
            return model_class(**params)
        return model_class()

    def train(self, train_df, kfolds, metric):
        val_scores, self.model_list = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            return_models=True
        )
        print(f"Val scores: {val_scores}")
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

        print(f"{metric.__name__}: {metric(y_data, predictions)}")
        for additional_metric in additional_metrics:
            print(
                f"{additional_metric.__name__}: {additional_metric(y_data, predictions)}")

        return predictions


class BoostingMixing:
    def find_hyperparameters(self, train_df, kfolds, metric, target=None):
        def try_hyperparameters(params):
            params['gpu_id'] = 0
            params['tree_method'] = 'gpu_hist'
            model = self.model_class(**params)
            return -1 * cross_validate(
                model,
                train_df.copy(),
                kfolds,
                metric,
                test_df=None,
            )

        result = fmin(
            fn=try_hyperparameters,
            space=self.tune_params,
            trials=Trials(),
            algo=tpe.suggest,
            max_evals=N_ITERS,
            return_argmin=False,
        )
        print(result)


class SklearnModelMixing:
    def find_hyperparameters(self, train_df, kfolds, metric, target=None):
        metric = prepare_wrap_metric(metric)
        splits = kfolds.split(train_df)
        search = RandomizedSearchCV(
            self.model, self.tune_params, cv=splits, scoring=metric,
            return_train_score=True, verbose=True, n_jobs=-1, n_iter=N_ITERS
        )

        search.fit(train_df.drop(target, axis=1), train_df[target])
        print(search.best_params_, search.best_score_)


class XGBoostModel(BaseModel, BoostingMixing):
    @staticmethod
    def get_tune_params():
        return {
            'max_depth': hp.choice('max_depth', range(2, 20, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.001, 0.5, 0.01),
            'n_estimators': hp.choice('n_estimators', range(100, 1000, 5)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'alpha': hp.uniform('alpha', 0, 80),
            'lambda': hp.uniform('lambda', 0, 10)
        }


class LRModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {}


class ENetModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {
            'alpha': np.arange(1e-4, 1e-3, 1e-4),
            'l1_ratio': np.arange(0.1, 1.0, 0.1),
            'max_iter': [1000]
        }


class LassoModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {
            'alpha': np.logspace(-4, -0.5, 30)
        }


class RidgeModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {
            'alpha': np.logspace(-4, -0.5, 30)
        }


class LSVRModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {
            'C': [0.1, 1, 10, 100, 1000],
        }


class SVRModel(BaseModel, SklearnModelMixing):
    @staticmethod
    def get_tune_params():
        return {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }


full_models = {
    "xgboost": XGBoostModel,
    "lr": LRModel,
    "e_net": ENetModel,
    "lasso": LassoModel,
    "ridge": RidgeModel,
    "l_svr": LSVRModel,
    "svr": SVRModel
}
