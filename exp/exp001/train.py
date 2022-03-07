import os
import gc
import re
import ast
import sys
import yaml
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings

from src.visualize import plot_importance
warnings.filterwarnings("ignore")
import logzero
import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from src.models import LGBModel, XGBModel, CBModel
from src.visualize import plot_importance
import lightgbm as lgb 
from lightgbm import early_stopping, log_evaluation, register_logger
import wandb 
from wandb.lightgbm import wandb_callback, log_summary
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error


def train_pipeline(cv, train, test, feature_cols, target_col, categorical_cols, config):
    oofs = []
    preds = []
    for fold, (idx_train, idx_valid) in enumerate(cv):
        logger.info("############")
        logger.info(f"fold {fold}")
        logger.info("############")

        train_data = train.loc[idx_train]
        valid_data = train.loc[idx_valid]
        X_train = train_data[feature_cols]
        X_valid = valid_data[feature_cols]
        y_train = train_data[target_col]
        y_valid = train_data[target_col]
        X_test = test[feature_cols]
        
        model = LGBModel(fold, config.model.params, logger, categorical_cols)
        model.train(X_train, y_train, X_valid, y_valid)
        model.save(OUTPUT_DIR, fold)
        pred = model.predict(X_valid)
        
        # evaluate
        score = mean_squared_error(y_true=valid_data[target_col], y_pred=pred)
        logger.info(f'fold-{fold} score: {score}')
        
        # create oof
        oof_df = pd.DataFrame(pred, index=idx_valid)
        oofs.append(oof_df)

        # pred
        pred_test = model.predict(X_test)
        np.save(OUTPUT_DIR + f"pred_test_{fold}", pred_test)
        preds.append(pred_test)
        
    # oofを保存
    oof = np.array(pd.concat(oofs).sort_index())
    np.save(oof, OUTPUT_DIR + "oof")

    # 全部のfoldの予測結果を保存
    pred_all = np.mean(preds, axis=1)
    np.save(pred_all, OUTPUT_DIR + "pred")
    return model 


@hydra.main(config_path='.', config_name='config')
def main(config: DictConfig):

    global OUTPUT_DIR, SEED, DEBUG, logger
    INPUT_DIR = config['global']['input_dir']
    OUTPUT_DIR = os.getcwd() + "/" # hydraが自動生成するoutputフォルダを指定
    SEED = config['global']['seed']
    DEBUG = config['global']['debug']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logzero.setup_logger(name='main', logfile=OUTPUT_DIR+'result.log', level=10)

    if not DEBUG:
        wandb.init(**config.wandb.init)
        wandb.config.version = config['global']['version']
        wandb.config.comment = config['global']['comment']
    
    ############
    # read data
    ############
    train = pd.read_csv(INPUT_DIR + 'test.csv')
    test = pd.read_csv(INPUT_DIR + 'train.csv')

    ############
    # feature engineering
    ############
    target_col = 'target'
    feature_cols = []
    categorical_cols = []

    ############
    # CV
    ############
    Fold = hydra.utils.instantiate(config.cv)
    cv = list(Fold.split(train, train[target_col]))

    ##########
    # train & predict
    ##########
    model = train_pipeline(train, test, cv, feature_cols, target_col, categorical_cols, config)
    plot_importance(model.models, train[feature_cols], OUTPUT_DIR)


if __name__=='__main__':
    main()