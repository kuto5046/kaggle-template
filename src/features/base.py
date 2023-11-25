import inspect
import time
from abc import ABCMeta
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.common import reduce_mem_usage


def get_categorical_col(df: pd.DataFrame, skip_cols: list = []):
    """カテゴリ型のカラム名を取得"""
    category_cols = []
    numerics = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]
    for col in df.columns:
        if col in skip_cols:
            continue
        col_type = df[col].dtypes
        if col_type not in numerics:
            category_cols.append(col)
    return category_cols


def get_numerical_col(df: pd.DataFrame, skip_cols: list = []):
    """数値型のカラム名を取得
    skip_colsにはtargetなど特徴量に追加しないものを選択
    """
    num_cols = []
    numerics = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]
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
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class Feature(metaclass=ABCMeta):
    prefix = ""
    suffix = ""
    dir = "/home/user/work/features/"
    fold_list = [0, 1, 2, 3, 4]  # 必要に応じて外から上書き
    """
    dirとfold listは以下のように外から上書きする
    Feature.dir = c.feature_dir
    Feature.fold_list = fold_list  # 特徴量を作成するfoldを指定
    """

    def __init__(self):
        self.reset()

    def run(self):
        prefix = self.prefix + "_" if self.prefix else ""
        suffix = "_" + self.suffix if self.suffix else ""
        with timer(self.name):
            # train
            # こちらは検証時foldごとに作成する特徴量
            # 保存するファイル名にfoldが含まれている
            for fold in self.fold_list:
                self.create_features(fold=fold)

                if self.train.shape[0] > 0:
                    self.train.columns = prefix + self.train.columns + suffix
                    train_path_fold = self.train_path.with_name(
                        f"{self.train_path.stem}_fold{fold}"
                    )
                    self.train = reduce_mem_usage(self.train)
                    self.train.to_pickle(f"{str(train_path_fold)}.pickle")

                if self.valid.shape[0] > 0:
                    self.valid.columns = prefix + self.valid.columns + suffix
                    valid_path_fold = self.valid_path.with_name(
                        f"{self.valid_path.stem}_fold{fold}"
                    )
                    self.valid = reduce_mem_usage(self.valid)
                    self.valid.to_pickle(f"{str(valid_path_fold)}.pickle")

                # 基本このphaseではself.testは作成しないが
                # embeddingの次元を揃える必要がある場合は
                # foldごとにtestの特徴量を作成する必要があるためself.testを作成する
                if self.test.shape[0] > 0:
                    self.test.columns = prefix + self.test.columns + suffix
                    test_path_fold = self.test_path.with_name(f"{self.test_path.stem}_fold{fold}")
                    self.test = reduce_mem_usage(self.test)
                    self.test.to_pickle(f"{str(test_path_fold)}.pickle")

                self.reset()

            # sub用(foldを切らずtrain全体で学習する場合にtrainとtestの特徴量を作成する)
            # こちらは保存するファイル名にfoldが含まれないことで上記のデータと見分けている
            self.create_features()
            if self.train.shape[0] > 0:
                self.train.columns = prefix + self.train.columns + suffix
                self.train = reduce_mem_usage(self.train)
                self.train.to_pickle(f"{str(self.train_path)}.pickle")
            if self.test.shape[0] > 0:
                self.test.columns = prefix + self.test.columns + suffix
                self.test = reduce_mem_usage(self.test)
                self.test.to_pickle(f"{str(self.test_path)}.pickle")
        return self

    def reset(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.valid = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f"{self.name}_train"
        self.valid_path = Path(self.dir) / f"{self.name}_valid"
        self.test_path = Path(self.dir) / f"{self.name}_test"

    @abstractmethod
    def create_features(self, fold: Optional[int] = None):
        """
        if fold is not None:
            # for cv
            _train, _valid = split_train_valid(train, fold)
            self.train = _train[col].to_numpy()
            self.valid = _valid[col].to_numpy()
        else:
            # for submission
            self.train = train[col].to_numpy()
            self.test = test[col].to_numpy()
        """
        raise NotImplementedError


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite: bool = False):
    for f in get_features(namespace):
        # foldごとのチェックが面倒なので一番最後に作成されるtestが存在するかで判定する
        if f.test_path.with_suffix(".pickle").exists() and not overwrite:
            print(f.name, "was skipped")
        else:
            f.run()


def remove_features(feature_dir: Path):
    """
    作成済みの特徴量をすべて削除する
    """
    for filepath in list(feature_dir.glob("*")):
        if filepath.is_file():
            filepath.unlink()


def load_datasets(
    feats: list[str],
    input_dir: Path = Path("/home/user/work/features/"),
    phase: str = "train",
    fold: Optional[int] = None,
) -> pd.DataFrame:
    """
    testをfoldで区切る必要があるケースがややこしい
    foldを指定してtestを読む場合まずfoldごとのデータがある前提で読みにいく。ない場合はfoldなしのデータを読む
    """
    assert phase in ["train", "valid", "test"]
    if phase == "train":
        if fold is None:
            # foldを分割していないtrain全体の特徴量
            dfs = [pd.read_pickle(input_dir / f"{f}_train.pickle") for f in feats]
        else:
            dfs = [pd.read_pickle(input_dir / f"{f}_train_fold{fold}.pickle") for f in feats]
    elif phase == "valid":
        dfs = [pd.read_pickle(input_dir / f"{f}_valid_fold{fold}.pickle") for f in feats]
    else:
        if fold is None:
            # foldに関係ないtestの特徴量
            dfs = [pd.read_pickle(input_dir / f"{f}_test_fold{fold}.pickle") for f in feats]
        else:
            # foldごとに作成しているtestの特徴量
            dfs = []
            for f in feats:
                if (input_dir / f"{f}_test_fold{fold}.pickle").exists():
                    dfs.append(pd.read_pickle(input_dir / f"{f}_test_fold{fold}.pickle"))
                else:
                    dfs.append(pd.read_pickle(input_dir / f"{f}_test.pickle"))

    X = pd.concat(dfs, axis=1)
    return X
