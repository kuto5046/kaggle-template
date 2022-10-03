# https://github.com/takapy0210/takaggle/blob/master/takaggle/training/model.py
import pickle
from abc import ABCMeta, abstractmethod
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from lightgbm import early_stopping, log_evaluation, register_logger
from wandb.lightgbm import wandb_callback as lgb_wandb_callback
from wandb.xgboost import wandb_callback as xgb_wandb_callback
from xgboost.callback import EarlyStopping, EvaluationMonitor


def get_model(model_name: str, model_params: dict, num_boost_round: int, cat_cols: list, output_dir: str, callbacks:list):
    """
    usage:
    model = get_model(model_name, model_params, num_boost_round, cat_cols, output_dir, callbacks=[])
    """
    if model_name=='lightgbm':
        model = LGBModel(model_params, num_boost_round, cat_cols, output_dir, callbacks)
    elif model_name=='xgboost':
        model = XGBModel(model_params, num_boost_round, cat_cols, output_dir, callbacks)
    elif model_name == 'catboost-regressor':
        model = CBRegressorModel(model_params, num_boost_round, cat_cols, output_dir, callbacks)
    elif model_name == 'catboost-classifier':
        model = CBClassifierModel(model_params, num_boost_round, cat_cols, output_dir, callbacks)
    else:
        NotImplementedError
    
    return model 


def get_callbacks(model_name):
    if model_name == 'lightgbm':
        callbacks = [
            early_stopping(stopping_rounds=100, first_metric_only=False, verbose=True),
            log_evaluation(period=100, show_stdv=True),
            lgb_wandb_callback()
        ]
    elif model_name == 'xgboost':
        callbacks = [
            EarlyStopping(rounds=100, data_name='valid'),
            EvaluationMonitor(period=100),
            xgb_wandb_callback()
        ]
    elif 'catboost' in model_name:
        callbacks = []
    return callbacks 

# def get_callbacks(config):
#     callbacks = []
#     if "callbacks" in config:
#         for _, cb_conf in config.callbacks.items():
#             if "_target_" in cb_conf:
#                 if ('wandb' in cb_conf._target_)&(wandb.run is not None):
#                     callbacks.append(hydra.utils.instantiate(cb_conf))
#                 else:
#                     callbacks.append(hydra.utils.instantiate(cb_conf))
#     return callbacks 

class BaseModel(metaclass=ABCMeta):

    def __init__(self, model_params: dict, num_boost_round: int, cat_cols: list=[], output_dir:str='./', callbacks: list=[]) -> None:

        self.fold = None 
        self.num_boost_round = num_boost_round
        self.model_params = model_params
        self.models = []
        self.cat_cols = cat_cols
        self.output_dir = output_dir
        self.callbacks = callbacks
        self.logger = None 

    def set_logger(self, logger):
        self.logger = logger

    def set_fold(self, fold):
        self.fold = fold 
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_valid: Optional[pd.DataFrame] = None,
              y_valid: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する"""
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.array:
        """学習済のモデルでの予測値を返す"""
        pass


    def save(self, fold) -> None:
        """モデルの保存を行う"""
        pickle.dump(self.model, open(f"{self.output_dir}/model_{fold}.pkl" , 'wb'))


    def load(self, fold) -> None:
        """モデルの読み込みを行う"""
        return pickle.load(open(f"{self.output_dir}/model_{fold}.pkl", 'rb'))


class LGBModel(BaseModel):

    def train(self, X_train, y_train, X_valid, y_valid):
        # データのセット
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.cat_cols)
        valid_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cat_cols)

        if self.logger is not None:
            register_logger(self.logger)  # custom loggerに学習のlogを流すようにする

        self.model = lgb.train(
            params=self.model_params, 
            train_set=train_set, 
            valid_sets=[train_set, valid_set], 
            num_boost_round=self.num_boost_round, 
            callbacks=self.callbacks,
            )
        # log_summary(self.model, feature_importance=False, save_model_checkpoint=False)
        self.models.append(self.model)

    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration)


class XGBModel(BaseModel):

    def train(self, X_train, y_train, X_valid, y_valid):

        # データのセット
        train_set = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        valid_set = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        # 学習
        self.model = xgb.train(
            self.model_params,
            train_set,
            num_boost_round=self.num_boost_round,
            evals=[(train_set, 'train'), (valid_set, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=100,
            )
        
        self.models.append(self.model)

    def predict(self, X_test):
        test_data = xgb.DMatrix(X_test, enable_categorical=True)
        return self.model.predict(test_data, iteration_range=(0, self.model.best_iteration + 1))


class CBRegressorModel(BaseModel):

    def train(self, X_train, y_train, X_valid, y_valid):

        self.model = CatBoostRegressor(**self.model_params)
        # category typeをintに変換
        X_train[self.cat_cols] = X_train[self.cat_cols].astype(int)
        X_valid[self.cat_cols] = X_valid[self.cat_cols].astype(int)
        cat_idx_cols = [X_train.columns.get_loc(col) for col in self.cat_cols]

        # 学習
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            metric_period=100,
            use_best_model=True,
            cat_features=cat_idx_cols
        )
        self.models.append(self.model)

    def predict(self, X_test):
        # category typeをintに変換
        X_test[self.cat_cols] = X_test[self.cat_cols].astype(int)
        return self.model.predict(X_test)


class CBClassifierModel(BaseModel):

    def train(self, X_train, y_train, X_valid, y_valid):

        self.model = CatBoostClassifier(**self.model_params)
        
        # category typeをintに変換
        X_train[self.cat_cols] = X_train[self.cat_cols].astype(int)
        X_valid[self.cat_cols] = X_valid[self.cat_cols].astype(int)
        cat_idx_cols = [X_train.columns.get_loc(col) for col in self.cat_cols]

        # 学習
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            metric_period=100,
            use_best_model=True,
            cat_features=cat_idx_cols
        )
        self.models.append(self.model)

    def predict(self, X_test):
        # category typeをintに変換
        X_test[self.cat_cols] = X_test[self.cat_cols].astype(int)
        return self.model.predict(X_test)
