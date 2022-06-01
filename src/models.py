# https://github.com/takapy0210/takaggle/blob/master/takaggle/training/model.py
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional
import logzero
from wandb.lightgbm import wandb_callback, log_summary
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, register_logger
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import os
import pickle 
import sys 
import hydra 
from omegaconf import DictConfig
from logging import Logger
import wandb 


def get_model(config: DictConfig, logger: Logger, cat_cols: list, output_dir: str, callbacks:list):
    """
    usage:
    model = get_model(config, fold, logger, cat_cols)
    """
    
    if config.model.name == 'lightgbm':
        model = LGBModel(config.model, logger, cat_cols, output_dir, callbacks)
    elif config.model.name == 'xgboost':
        model = XGBModel(config.model, logger, cat_cols, output_dir)
    elif config.model.name == 'catboost':
        model = CBModel(config.model, logger, cat_cols, output_dir)
    else:
        NotImplementedError
    
    return model 

def get_callbacks(config):
    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                if ('wandb' in cb_conf._target_)&(wandb.run is not None):
                    callbacks.append(hydra.utils.instantiate(cb_conf))
                else:
                    callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks 


class BaseModel(metaclass=ABCMeta):

    def __init__(self, model_config: dict, logger: Logger, cat_cols: list=[], output_dir:str='./', callbacks: list=[]) -> None:

        self.fold = None 
        self.num_round = model_config.num_boost_round
        self.params = dict(model_config.params)
        self.model = None
        self.models = []
        self.cat_cols = cat_cols
        self.logger = logger
        self.output_dir = output_dir
        self.callbacks = callbacks

    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        :param tr_x: 学習データの特徴量
        :param tr_y: 学習データの目的変数
        :param va_x: バリデーションデータの特徴量
        :param va_y: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass


    def save(self, fold) -> None:
        """モデルの保存を行う"""
        pickle.dump(self.model, open(self.output_dir + f"model_{fold}.pkl" , 'wb'))


    def load(self, fold) -> None:
        """モデルの読み込みを行う"""
        pickle.load(open(self.output_dir + f"model_{fold}.pkl", 'rb'))


class LGBModel(BaseModel):

    def train(self, tr_x, tr_y, va_x, va_y):
        # データのセット
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=self.cat_cols)
        valid_set = lgb.Dataset(va_x, va_y, categorical_feature=self.cat_cols)
        register_logger(self.logger)  # custom loggerに学習のlogを流すようにする
        self.model = lgb.train(
            params=self.params, 
            train_set=train_set, 
            valid_sets=[train_set, valid_set], 
            num_boost_round=self.num_round, 
            callbacks=self.callbacks,
            )
        if wandb.run is not None:
            log_summary(self.model, feature_importance=False, save_model_checkpoint=False)
        self.models.append(self.model)

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


class XGBModel(BaseModel):
    def __init__(self, fold: int, params: dict, logger: Logger, cat_cols: list=[], output_dir:str='./') -> None:
        super().__init__(fold, params, logger, cat_cols, output_dir)

    def train(self, tr_x, tr_y, va_x, va_y):

        # データのセット
        train_set = xgb.DMatrix(tr_x, label=tr_y)
        valid_set = xgb.DMatrix(va_x, label=va_y)

        # 学習
        watchlist = [(train_set, 'train'), (valid_set, 'eval')]
        self.model = xgb.train(
            dict(self.params),
            train_set,
            num_round=10000,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=100,
            )
        if wandb.run is not None:
            log_summary(self.model, feature_importance=True, save_model_checkpoint=True)
        self.models.append(self.model)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)


class CBModel(BaseModel):
    def __init__(self, fold: int, params: dict, logger: Logger, cat_cols: list=[], output_dir:str='./') -> None:
        super().__init__(fold, params, logger, cat_cols, output_dir)
        self.pred_type = None  # catboostの回帰or分類タイプ

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # ハイパーパラメータの設定
        params = dict(self.params)
        self.pred_type = params.pop('pred_type')

        if self.pred_type == 'Regressor':
            self.model = CatBoostRegressor(**params)
        elif self.pred_type == 'Classifier':
            self.model = CatBoostClassifier(**params)
        else:
            print('pred_typeが正しくないため終了します')
            sys.exit(0)

        # 学習
        self.model.fit(
            tr_x,
            tr_y,
            eval_set=[(va_x, va_y)],
            verbose=100,
            use_best_model=True,
            cat_features=self.cat_cols
        )

    def predict(self, te_x):
        if self.pred_type == 'Regressor':
            return self.model.predict(te_x)
        elif self.pred_type == 'Classifier':
            return self.model.predict_proba(te_x)[:, 1]
        return self.model.predict(te_x)
