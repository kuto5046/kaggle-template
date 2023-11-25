import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold

# from sklearn.utils import _deprecate_positional_args


def split_train_valid(train: pd.DataFrame, fold: int):
    """
    与えられたfoldをもとにtrainをtrainとvalidに分割する
    """
    valid_fold = [fold]
    _train = train.query("fold not in @valid_fold").reset_index(drop=True).copy()
    _valid = train.query("fold in @valid_fold").reset_index(drop=True).copy()
    return _train, _valid


def get_fold(_train: pd.DataFrame, cv: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    trainにfoldのcolumnを付与する
    """
    train = _train.copy()
    train["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(cv):
        train.loc[valid_idx, "fold"] = fold
    print(train["fold"].value_counts())
    return train


def get_kfold(train: pd.DataFrame, n_splits: int, seed: int = 0) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train))
    return get_fold(train, cv)


def get_stratifiedkfold(
    train: pd.DataFrame, target_col: str, n_splits: int, seed: int = 0
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(skf.split(X=train, y=train[target_col]))
    return get_fold(train, cv)


def get_groupkfold(
    train: pd.DataFrame, target_col: str, group_col: str, n_splits: int
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    cv = list(gkf.split(X=train, y=train[target_col], groups=train[group_col]))
    return get_fold(train, cv)


def get_stratifiedgroupkfold(
    train: pd.DataFrame, target_col: str, group_col: str, n_splits: int, seed=0
) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=seed)
    cv = list(sgkf.split(X=train, y=train[target_col], groups=train[group_col]))
    return get_fold(train, cv)
