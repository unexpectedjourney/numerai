import numpy as np
from catboost import CatBoostRegressor
from hyperopt import Trials, tpe, fmin
from hyperopt import hp
from lightgbm import LGBMRegressor
from scipy.stats import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor

from src.train_method import cross_validate
from src.utils import (
    prepare_wrap_metric, create_era_correlation, plot_era_corr)
from src.metrics import spearmanr, smart_sharpe

N_ITERS = 20
models = {
    "xgboost": XGBRegressor,
    "lgboost": LGBMRegressor,
    "catboost": CatBoostRegressor,
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

    def train(self, train_df, kfolds, metric, era_metrics=None, plot_eras=False):
        if era_metrics is None:
            era_metrics = []

        val_scores, self.model_list, val_preds = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            return_models=True,
            return_val_preds=True,
            copy_model=True
        )

        era_scores = create_era_correlation(
            train_df.target.tolist(),
            val_preds.target.tolist(),
            train_df.era.tolist(),
            spearmanr
        )
        if plot_eras:
            plot_era_corr(era_scores, self.model)

        for era_metric in era_metrics:
            print(f"{era_metric.__name__}: {era_metric(era_scores)}")

        return self

    def predict(self, train_df, kfolds, test_df, target, metric):
        val_scores, predictions = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            test_df=test_df,
            copy_model=True
        )
        predictions = np.mean(np.array(predictions), axis=0)

        return predictions

    def predict_and_score(self, train_df, kfolds, test_df, target, metric,
                          additional_metrics=None, era_metrics=None,
                          plot_eras=False):
        if additional_metrics is None:
            additional_metrics = []

        if era_metrics is None:
            era_metrics = []

        assert target in test_df.columns, "Please specify target in test_df"

        y_data = test_df[target].values

        val_scores, predictions = cross_validate(
            self.model,
            train_df,
            kfolds,
            metric,
            test_df=test_df,
            copy_model=True
        )
        predictions = np.mean(np.array(predictions), axis=0)

        print(f"{metric.__name__}: {metric(y_data, predictions)}")
        for additional_metric in additional_metrics:
            print(f"{additional_metric.__name__}: {additional_metric(y_data, predictions)}")

        era_scores = create_era_correlation(
            test_df.target.tolist(),
            predictions,
            test_df.era.tolist(),
            spearmanr
        )

        if plot_eras:
            plot_era_corr(era_scores, self.model)

        for era_metric in era_metrics:
            print(f"{era_metric.__name__}: {era_metric(era_scores)}")

        return predictions


class BoostingMixing:
    def find_hyperparameters(self, train_df, kfolds, metric, target=None):
        def try_hyperparameters(params):
            model = self.model_class(**params)
            val_scores, predictions = cross_validate(
                model,
                train_df.copy(),
                kfolds,
                metric,
                test_df=None,
                copy_model=True,
                return_val_preds=True,
            )

            # era_scores = create_era_correlation(
            #     train_df.target.tolist(),
            #     predictions.target.tolist(),
            #     train_df.era.tolist(),
            #     spearmanr
            # )
            # score = smart_sharpe(era_scores)
            # return -score
            return -val_scores

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
            'gpu_id': 0,
            'tree_method': 'gpu_hist',
            'booster': hp.choice('booster', ('gbtree', 'gblinear', 'dart')),
            'max_depth': hp.choice('max_depth', range(2, 20, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.001, 0.5, 0.01),
            'n_estimators': hp.choice('n_estimators', range(1000, 4001, 25)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'alpha': hp.uniform('alpha', 0, 80),
            'lambda': hp.uniform('lambda', 0, 10)
        }


class LGBoostModel(BaseModel, BoostingMixing):
    @staticmethod
    def get_tune_params():
        return {
            'device': 'gpu',
            'boosting_type': hp.choice('boosting_type',
                                       ['dart', 'gbdt', 'goss']),
            'learning_rate': hp.choice('learning_rate',
                                       np.arange(0.005, 0.1005, 0.005)),
            'n_estimators': hp.choice('n_estimators',
                                      np.arange(1000, 4001, 25, dtype=int)),
            'max_depth': hp.choice('max_depth',
                                   np.arange(5, 70, 2, dtype=int)),
            'num_leaves': hp.choice('num_leaves',
                                    [3, 5, 7, 15, 31, 50, 75, 100]),
            'feature_fraction': hp.uniform('feature_fraction', 0, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.),
            'lambda_l1': hp.loguniform('lambda_l1', -3, 2),
            'lambda_l2': hp.loguniform('lambda_l2', -3, 2),
        }


class CatBoostModel(BaseModel, BoostingMixing):
    @staticmethod
    def get_tune_params():
        return {
            'depth': hp.quniform('depth', 2, 16, 1),
            'max_bin': hp.quniform('max_bin', 1, 32, 1),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 5),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
            'random_strength': hp.loguniform('random_strength', np.log(0.005),
                                             np.log(5)),
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.25),
            'fold_len_multiplier': hp.loguniform('fold_len_multiplier',
                                                 np.log(1.01), np.log(2.5)),
            'od_type': 'Iter',
            'od_wait': 25,
            'task_type': 'GPU',
            'devices': '0:1',
            'verbose': 0
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
    "lgboost": LGBoostModel,
    "catboost": CatBoostModel,
    "lr": LRModel,
    "e_net": ENetModel,
    "lasso": LassoModel,
    "ridge": RidgeModel,
    "l_svr": LSVRModel,
    "svr": SVRModel
}
