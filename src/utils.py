# Gistから必要なスニペットを取ってくる
# https://gist.github.com/kuto5046
import json
import os
import subprocess
import sys
from datetime import datetime
import builtins
import types
import random 
import torch
import hydra
import numpy as np
import pandas as pd
import requests
import pickle 
import logging
from pathlib import Path
from omegaconf import OmegaConf
# .env ファイルをロードして環境変数へ反映
# from dotenv import load_dotenv

# load_dotenv()


class HydraConfig():
    """
    hydraによる設定値の取得
    scriptの時はデコレータの方が簡単なのでこちらは使わずnotebookの時に使用することを想定

    usage:
    model_config = HydraConfig.get_cnf(config_dir=Path('config/model'), config_name='default')
    """
    @staticmethod
    def get_cnf(config_dir: Path | str, config_name: str) -> OmegaConf:
        # 絶対パスに変換
        if isinstance(config_dir, str):
            config_dir = os.path.join(os.getcwd(), config_dir)
        elif isinstance(config_dir, Path):
            config_dir = str(config_dir.absolute())
        else:
            raise ValueError(f"config_dir is not str or Path. config_dir: {config_dir}")

        if not os.path.isdir(config_dir):
            print(f"Can not find file: {config_dir}.")
            sys.exit(-1)
        
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None, job_name='exp'):
            cnf = hydra.compose(config_name=config_name)
            return cnf

    
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type not in [object, 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def get_hash(config):
    """get git hash value(short ver.)
    """
    if config['globals']["kaggle"]:
        # kaggle
        hash_value = None
    else:
        # local
        cmd = "git rev-parse --short HEAD"
        hash_value = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    
    return hash_value


def get_timestamp():
    # output config
    return datetime.today().strftime("%m%d-%H%-M%-S")


# 任意のメッセージを通知する関数
def send_slack_message_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']  
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)

# errorを通知する関数
def send_slack_error_notification(message):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']
    # no_entry_signは行き止まりの絵文字を出力
    data = json.dumps({"text":":no_entry_sign:" + message})  
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def pickle_save(object: object, path: Path):
    pickle.dump(object, open(path, 'wb'))


def pickle_load(path: Path):
    return pickle.load(open(path, 'rb'))


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def get_logger(output_dir: Path, file_name: str = 'result.log'):
    """ 
    from src.utils import get_logger
    logger = get_logger(output_dir)
    """
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    # File handler for outputting to a log file
    file_handler = logging.FileHandler(output_dir / file_name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler for outputting to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Redirect stdout and stderr
    # sys.stdout = StreamToLogger(logger, logging.INFO)
    # sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger 
