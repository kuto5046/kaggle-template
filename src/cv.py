import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
# from sklearn.utils import _deprecate_positional_args  


def split_train_valid(train: pd.DataFrame, fold: int):
    """
    与えられたfoldをもとにtrainをtrainとvalidに分割する
    """
    valid_fold = [fold]
    _train = train.query('fold not in @valid_fold').reset_index(drop=True).copy()
    _valid = train.query('fold in @valid_fold').reset_index(drop=True).copy()
    return _train, _valid


def get_fold(_train: pd.DataFrame, cv:list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    trainにfoldのcolumnを付与する
    """
    train = _train.copy()
    train['fold'] = -1
    for fold, (train_idx, valid_idx) in enumerate(cv):
        train.loc[valid_idx, 'fold'] = fold
    print(train['fold'].value_counts())
    return train


def get_kfold(train: pd.DataFrame, n_splits: int, seed:int=0) -> pd.DataFrame:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(kf.split(X=train))
    return get_fold(train, cv)


def get_stratifiedkfold(train: pd.DataFrame, target_col: str, n_splits: int, seed: int=0) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = list(skf.split(X=train, y=train[target_col]))
    return get_fold(train, cv)


def get_groupkfold(train: pd.DataFrame, target_col: str, group_col: str, n_splits: int) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    cv = list(gkf.split(X=train, y=train[target_col], groups=train[group_col]))
    return get_fold(train, cv)


def get_stratifiedgroupkfold(train: pd.DataFrame, target_col: str, group_col: str, n_splits: int, seed=0) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=seed)
    cv = list(sgkf.split(X=train, y=train[target_col], groups=train[group_col]))
    return get_fold(train, cv)


# def get_cont_stratifiedkfold(train, target_col, n_splits, q=10, seed=0) -> pd.DataFrame:
#     gkf = ContinuousStratifiedFold(n_splits=n_splits, q=q, shuffle=True, random_state=seed)
#     cv = list(gkf.split(X=train, y=train[target_col]))
#     return get_fold(train, cv)


# class ContinuousStratifiedFold(StratifiedKFold):
#     """
#     stratified-K-Fold Splits for continuous target.
#     """

#     @_deprecate_positional_args
#     def __init__(self, n_splits=5, *, q=10, shuffle=False, random_state=None):
#         """
#         Args:
#             n_splits:
#                 number of splits
#             q:
#                 number of quantiles.
#                 例えば10に設定されると, `y` を値に応じて 10 個の集合に分割し, 各 fold では train / valid に集合ごとの
#                 割合が同じように選択されます.
#             shuffle:
#                 True の時系列をシャッフルしてから分割します
#             random_state:
#                 シャッフル時のシード値
#         Notes:
#             y の水準数 (i.e. uniqueな数値の数) よりも大きい値を設定すると集合の分割に失敗し split を実行する際に
#             ValueError となることがあります. その場合小さい `q` を設定することを検討してください
#         """
#         super().__init__(n_splits=n_splits, shuffle=shuffle,
#                          random_state=random_state)
#         self.q = q

#     def split(self, X, y, groups=None):
#         try:
#             y_cat = pd.qcut(y, q=self.q).values.codes
#         except ValueError as e:
#             raise ValueError('Fail to quantile cutting (using pandas.qcut). '
#                              'There are cases where this value fails when you make it too large') from e
#         return super(ContinuousStratifiedFold, self).split(X, y_cat, groups=groups)