import pandas as pd 
import hydra
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.utils import _deprecate_positional_args  


def get_kfold(train, n_splits, seed=0):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(fold.split(X=train))


def get_stratifiedkfold(train, target_col, n_splits, seed=0):
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(fold.split(X=train, y=train[target_col]))


def get_groupkfold(train, target_col, group_col, n_splits):
    fold = GroupKFold(n_splits=n_splits)
    return list(fold.split(X=train, y=train[target_col], groups=train[group_col]))


def get_cont_stratifiedkfold(train, target_col, n_splits, q=10, seed=0):
    fold = ContinuousStratifiedFold(n_splits=n_splits, q=q, shuffle=True, random_state=seed)
    return list(fold.split(X=train, y=train[target_col]))


# hydraを想定
def get_fold(X, config, y=None, groups=None):
    """
    usage:
    cv = get_fold(train, train['target'], config)
    for (idx_train, idx_valid) in cv:
        break
    """
    fold = hydra.utils.instantiate(config.cv)
    fold_name = config.cv._target_.split('.')[-1]
    if fold_name == 'KFold':
        return list(fold.split(X=X))
    elif fold_name == 'StratifiedKFold':
        return list(fold.split(X=X, y=y, groups=groups))
    elif fold_name == 'GroupKFold':
        return list(fold.split(X=X, y=y, groups=groups))
    elif fold_name == 'ContinuousStratifiedFold':
        return list(fold.split(X=X, y=y, groups=groups))


class ContinuousStratifiedFold(StratifiedKFold):
    """
    stratified-K-Fold Splits for continuous target.
    """

    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, q=10, shuffle=False, random_state=None):
        """
        Args:
            n_splits:
                number of splits
            q:
                number of quantiles.
                例えば10に設定されると, `y` を値に応じて 10 個の集合に分割し, 各 fold では train / valid に集合ごとの
                割合が同じように選択されます.
            shuffle:
                True の時系列をシャッフルしてから分割します
            random_state:
                シャッフル時のシード値
        Notes:
            y の水準数 (i.e. uniqueな数値の数) よりも大きい値を設定すると集合の分割に失敗し split を実行する際に
            ValueError となることがあります. その場合小さい `q` を設定することを検討してください
        """
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)
        self.q = q

    def split(self, X, y, groups=None):
        try:
            y_cat = pd.qcut(y, q=self.q).values.codes
        except ValueError as e:
            raise ValueError('Fail to quantile cutting (using pandas.qcut). '
                             'There are cases where this value fails when you make it too large') from e
        return super(ContinuousStratifiedFold, self).split(X, y_cat, groups=groups)