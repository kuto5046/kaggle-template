from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd

from src.features.base import Feature
from src.features.base import generate_features
from src.features.base import get_categorical_col
from src.features.encoder import OrdinalEncoder
from src.features.nlp import UniversalSentenceEncoder
from src.utils.split import get_stratifiedkfold
from src.utils.split import split_train_valid

TARGET_COL = "is_kokuhou"


class Target(Feature):
    def create_features(self, train, test, fold: Optional[int] = None):
        if fold is not None:
            _train, _valid = split_train_valid(train, fold)
            self.train = _train[[TARGET_COL]]
            self.valid = _valid[[TARGET_COL]]
        else:
            # for submission
            self.train = train[[TARGET_COL]].copy()
            self.test = pd.DataFrame(np.zeros(len(test)), columns=[TARGET_COL])  # dummy


class OrdinalEncode(Feature):
    def ordinal_encoder(
        self, train: pd.DataFrame, test: pd.DataFrame, cols: list[str], prefix: str
    ):
        encoder = OrdinalEncoder()
        _whole = pd.concat([train, test], axis=0).reset_index(drop=True)
        encoder.fit(_whole[cols])
        _train = (
            pd.DataFrame(encoder.transform(train[cols]), columns=cols)
            .add_prefix(prefix)
            .astype("category")
        )
        _test = (
            pd.DataFrame(encoder.transform(test[cols]), columns=cols)
            .add_prefix(prefix)
            .astype("category")
        )
        return _train, _test

    def create_features(self, train, test, fold: Optional[int] = None):
        use_cols = get_categorical_col(self.train, skip_cols=["名称", "所在地"])
        prefix = "ordinal_enc_"
        _train, _test = self.ordinal_encoder(train, test, use_cols, prefix)

        if fold is not None:
            _train["fold"] = train["fold"].to_numpy()
            _train, _valid = split_train_valid(_train, fold)
            self.train = _train.filter(like=prefix)
            self.valid = _valid.filter(like=prefix)
        else:
            self.train = _train.copy()
            self.test = _test.copy()


class TextUSEncoderFeature(Feature):
    """
    日本語のカラムを全てconcatしてUSEで特徴量を作成する
    """

    def create_features(self, train, test, fold: int | None = None):
        use_cols = get_categorical_col(train, skip_cols=["名称"])
        text_col = "text"
        _train = train.fillna("").copy()
        _test = test.fillna("").copy()
        _train[text_col] = _train[use_cols[0]].str.cat(_train[use_cols[1:]], sep=" ")
        _test[text_col] = _test[use_cols[0]].str.cat(_test[use_cols[1:]], sep=" ")
        useencoder = UniversalSentenceEncoder()
        _train = useencoder.vectorize(_train, text_col, save=False)
        _test = useencoder.vectorize(_test, text_col, save=False)
        if fold is not None:
            _train["fold"] = train["fold"].to_numpy()
            _train, _valid = split_train_valid(_train, fold)
            self.train = _train.filter(like=text_col).copy()
            self.valid = _valid.filter(like=text_col).copy()
        else:
            self.train = _train.filter(like=text_col).copy()
            self.test = _test.filter(like=text_col).copy()


class NumericalFeature(Feature):
    def create_features(self, train, test, fold: Optional[int] = None):
        use_cols = [
            "緯度",
            "経度",
        ]
        if fold is not None:
            _train, _valid = split_train_valid(train, fold)
            self.train = _train[use_cols].copy()
            self.valid = _valid[use_cols].copy()
        else:
            self.train = train[use_cols].copy()
            self.test = test[use_cols].copy()


class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def read_data(self):
        self.train = pd.read_csv(Path(self.cfg.path.input_dir) / "train.csv")
        self.test = pd.read_csv(Path(self.cfg.path.input_dir) / "test.csv")

    def preprocess(self):
        pass

    def add_fold(self):
        self.train = get_stratifiedkfold(self.train, target_col=TARGET_COL, n_splits=5, seed=0)

    def run(self):
        self.read_data()
        self.preprocess()
        self.add_fold()
        Feature.dir = self.cfg.path.feature_dir
        generate_features(globals(), self.train, self.test, overwrite=self.cfg.overwrite)


@hydra.main(config_path="../conf", config_name="data_processor")
def main(cfg):
    data_processor = DataProcessor(cfg)
    data_processor.run()


if __name__ == "__main__":
    main()
