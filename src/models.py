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


class BaseModel(metaclass=ABCMeta):

    def __init__(self, fold: int, params: dict, logger, categoricals=[]) -> None:
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.fold = fold
        self.params = params
        self.model = None
        self.models = []
        self.categoricals = categoricals
        self.pred_type = None  # catboostの回帰or分類タイプ
        self.logger = logger

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


    def save(self, path, fold) -> None:
        """モデルの保存を行う"""
        pickle.dump(self.model, open(path + f"model_{fold}.pkl" , 'wb'))


    def load(self, path, fold) -> None:
        """モデルの読み込みを行う"""
        pickle.load(open(path + f"model_{fold}.pkl", 'rb'))


class LGBModel(BaseModel):

    def train(self, tr_x, tr_y, va_x, va_y):
        register_logger(self.logger)  # custom loggerに学習のlogを流すようにする
        callbacks = []
        callbacks.append(early_stopping(100))
        callbacks.append(log_evaluation(100))

        # データのセット
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=self.categoricals)
        valid_set = lgb.Dataset(va_x, va_y, categorical_feature=self.categoricals)
        register_logger(self.logger)  # custom loggerに学習のlogを流すようにする
        callbacks = []
        callbacks.append(early_stopping(100))
        callbacks.append(log_evaluation(100))
        callbacks.append(wandb_callback())
        self.model = lgb.train(
            params=dict(self.params), 
            train_set=train_set, 
            valid_sets=[train_set, valid_set], 
            num_boost_round=10000, 
            callbacks=callbacks,
            )
        log_summary(self.model, feature_importance=True, save_model_checkpoint=True)
        self.models.append(self.model)

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


class XGBModel(BaseModel):

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
        log_summary(self.model, feature_importance=True, save_model_checkpoint=True)
        self.models.append(self.model)

    def predict(self, te_x):
        dtest = xgb.DMatrix(te_x)
        return self.model.predict(dtest, ntree_limit=self.model.best_ntree_limit)


class CBModel(BaseModel):

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
            cat_features=self.categoricals
        )

    def predict(self, te_x):
        if self.pred_type == 'Regressor':
            return self.model.predict(te_x)
        elif self.pred_type == 'Classifier':
            return self.model.predict_proba(te_x)[:, 1]
        return self.model.predict(te_x)
