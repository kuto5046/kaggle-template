import os
import sys

# warnings.filterwarnings("ignore")
import logzero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from tqdm.auto import tqdm

sys.path.append("../../")
import hydra
import wandb
from omegaconf import DictConfig
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from src import utils as u
from src.cv import get_fold
from src.features.base import (get_categorical_col, get_numerical_col,
                               load_datasets)
from src.features.encoder import target_encoder
from src.models import get_callbacks, get_model
from src.utils import noglobal
from src.visualize import plot_importance


def calc_score(y_valid, pred):
    return roc_auc_score(y_true=y_valid, y_score=pred)


def train_pipeline(train, test, cv, config, cat_cols, target_col):
    oofs = []
    preds = []
    callbacks = get_callbacks(config)
    model = get_model(config, logger, cat_cols, OUTPUT_DIR, callbacks)
    for fold, (idx_train, idx_valid) in enumerate(cv):
        if (not DEBUG)&(config.logger.name=='wandb'):
            wandb.init(**config.logger.init, name=f'exp{config.exp.version}-fold{fold}')
            wandb.config = dict(config)

        logger.info("############")
        logger.info(f"fold {fold}")
        logger.info("############")

        _train = train.loc[idx_train].reset_index(drop=True)
        _valid = train.loc[idx_valid].reset_index(drop=True)

        # target encoding
        # for col in cat_cols:
        #     _train, _valid = target_encoder(_train, _valid, col, target_col)
        #     _, test = target_encoder(train, test, col, target_col)

        X_train = _train.drop(target_col, axis=1)
        y_train = _train[target_col]
        X_valid = _valid.drop(target_col, axis=1)
        y_valid = _valid[target_col]
        X_test = test

        model.train(X_train, y_train, X_valid, y_valid)
        model.save(fold)
        pred = model.predict(X_valid)

        # evaluate
        score = calc_score(y_valid, pred)
        logger.info(f'fold-{fold} score: {score}')
        wandb.log({'CV': score})

        # create oof
        oof_df = pd.DataFrame(pred, index=idx_valid)
        oofs.append(oof_df)

        # pred
        pred_test = model.predict(X_test)
        np.save(OUTPUT_DIR + f"pred_test_{fold}", pred_test)
        preds.append(pred_test)

        if (not DEBUG)&(config.logger.name=='wandb')&(fold!=len(cv)-1):
            wandb.finish()

    # oofを保存
    oof = np.array(pd.concat(oofs).sort_index())
    np.save(OUTPUT_DIR + "oof", oof)
    return model 



@hydra.main(config_path='../../configs/', config_name='config', version_base=None)
def main(config: DictConfig):

    global INPUT_DIR, OUTPUT_DIR, SEED, DEBUG, logger
    # config.common.timestamp = u.get_timestamp()
    INPUT_DIR = config.common.input_dir
    OUTPUT_DIR = config.common.output_dir
    SEED = config.common.seed
    DEBUG = config.common.debug
    target_col = config.common.target_col
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logzero.setup_logger(
        name='main', logfile=OUTPUT_DIR+'result.log', level=10)
    
    ############
    # read data
    ############
    feats = config.features
    train, test = load_datasets(feats, input_dir='/home/user/work/feature_store/')
    _train = pd.read_csv(INPUT_DIR + 'train.csv')
    train[target_col] = _train[target_col]
    cat_cols = get_categorical_col(train)

    ##########
    # train & predict
    ##########
    cv = get_fold(train, config, y=train[target_col])
    model = train_pipeline(train, test, cv, config, cat_cols, target_col)
    plot_importance(model.models, output_dir=OUTPUT_DIR)
    if wandb.run is None:
        wandb.log({'importance': wandb.Image(OUTPUT_DIR + f"importance.png")})
        wandb.alert(f'finish project:{wandb.project} group:{wandb.group}')
        wandb.finish()

    preds = []
    for i in range(len(cv)):
        pred = np.load(OUTPUT_DIR + f'pred_test_{i}.npy')
        preds.append(pred)
    pred_test = np.mean(preds, axis=0)

    sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
    sub['label'] = pred_test
    sub.to_csv(f'submission_exp{config.exp.version}.csv', index=False)

if __name__=='__main__':
    main()
