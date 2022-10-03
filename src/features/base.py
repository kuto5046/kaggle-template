import pandas as pd
from abc import ABCMeta, abstractmethod
import time 
from pathlib import Path
from contextlib import contextmanager
import inspect

def get_categorical_col(df:pd.DataFrame, skip_cols: list=[]):
    """カテゴリ型のカラム名を取得"""
    category_cols = []
    numerics = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64']
    for col in df.columns:
        if col in skip_cols:
            continue 
        col_type = df[col].dtypes
        if col_type not in numerics:
            category_cols.append(col)
    return category_cols


def get_numerical_col(df:pd.DataFrame, skip_cols: list=[]):
    """数値型のカラム名を取得
    skip_colsにはtargetなど特徴量に追加しないものを選択
    """
    num_cols = []
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        if col in skip_cols:
            continue 
        col_type = df[col].dtypes
        if col_type in numerics:
            num_cols.append(col)
    return num_cols


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '/home/user/work/feature_store/'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train'
        self.test_path = Path(self.dir) / f'{self.name}_test'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self, ext):
        if ext == 'csv':
            self.train.to_csv(f"{str(self.train_path)}.csv", index=False)
            self.test.to_csv(f"{str(self.test_path)}.csv", index=False)
        elif ext == 'parquet':
            self.train.to_parquet(f"{str(self.train_path)}.parquet")
            self.test.to_parquet(f"{str(self.test_path)}.parquet")
        elif ext == 'pickle':
            self.train.to_pickle(f"{str(self.train_path)}.pickle")
            self.test.to_pickle(f"{str(self.test_path)}.pickle")
        else:
            ValueError


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, ext: str = 'pickle', overwrite: bool =False):
    """parquetだとcategoryがfloatになってしまう"""
    for f in get_features(namespace):
        if f.train_path.with_suffix(f".{ext}").exists() and f.test_path.with_suffix(f".{ext}").exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save(ext)


def load_datasets(feats, input_dir = '/home/user/work/feature_store/', ext='pickle'):
    if ext == 'csv':
        train_dfs = [pd.read_csv(f'{input_dir}/{f}_train.csv') for f in feats]
        test_dfs = [pd.read_csv(f'{input_dir}/{f}_test.csv') for f in feats]
    elif ext == 'parquet':
        train_dfs = [pd.read_parquet(f'{input_dir}/{f}_train.parquet') for f in feats]
        test_dfs = [pd.read_parquet(f'{input_dir}/{f}_test.parquet') for f in feats]
    elif ext == 'pickle':
        train_dfs = [pd.read_pickle(f'{input_dir}/{f}_train.pickle') for f in feats]
        test_dfs = [pd.read_pickle(f'{input_dir}/{f}_test.pickle') for f in feats]
    else:
        ValueError

    X_train = pd.concat(train_dfs, axis=1)
    X_test = pd.concat(test_dfs, axis=1)
    assert X_train.shape[1]==X_test.shape[1]
    return X_train, X_test