import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MultiLabelBinarizer
import numpy as np 
from typing import List
from sklearn.model_selection import KFold

def get_non_overlapping(train: pd.DataFrame, test: pd.DataFrame, col: str):
    """
    train/testにしか出てこない値を調べる
    """
    only_in_train = set(train[col].unique()) - set(test[col].unique())
    only_in_test = set(test[col].unique()) - set(train[col].unique())
    non_overlapping = only_in_train.union(only_in_test)
    return non_overlapping


def pp_for_categorical_encoding(train: pd.DataFrame, test: pd.DataFrame, cols:List[str]):
    """
    encoding対象のcolumnに欠損がある場合は欠損を埋める
    encoding対象のcolumnにtrain/testにしか出てこない値は全て同じものとして扱う
    ex)trainにのみ'A', testにのみ'B'というカテゴリがある場合, 'A'と'B'は'other'という一つの値として扱う
    """
    for col in cols:
        non_overlapping = get_non_overlapping(train, test, col)
        try:
            if train[col].dtype == np.dtype("O"):
                # dtypeがobjectなら欠損は'missing' クラスにする
                train[col] = train[col].fillna("missing")
                test[col] = test[col].fillna("missing")
                train[col] = train[col].map(lambda x: x if x not in non_overlapping else "other")
                test[col] = test[col].map(lambda x: x if x not in non_overlapping else "other")

            else:
                # dtypeがint/floatなら欠損は-1とする
                train[col] = train[col].fillna(-1)
                test[col] = test[col].fillna(-1)
                train[col] = train[col].map(lambda x: x if x not in non_overlapping else -2)
                test[col] = test[col].map(lambda x: x if x not in non_overlapping else -2)
        except:
            print(f"Error at {col} columns")
    return train, test

# encoder系はtaret encodingを除いて全て新しく作成されたcolumnのみを返す
def count_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    usage:
    ce_train, ce_test = count_encoder(train, test, cols=['cat1', 'cat2'])
    """
    train = train_df.copy()
    test = test_df.copy()
    prefix = 'count_enc'
    for col in cols:
        encoder = train[col].value_counts()
        train[f'{prefix}_{col}'] = train[col].map(encoder)
        test[f'{prefix}_{col}'] = test[col].map(encoder)
    train = train.filter(like=prefix)
    test = test.filter(like=prefix)
    return train, test


def ordinal_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    label_encoderと同じ挙動のはず
    usage:
    le_train, le_test = ordinal_encoder(train, test, cols=['cat1', 'cat2'])
    """
    train = train_df.copy()
    test = test_df.copy()
    
    encoder = OrdinalEncoder()
    _whole = pd.concat([train, test], axis=0).reset_index(drop=True)
    encoder.fit(_whole[cols])
    _train = pd.DataFrame(encoder.transform(train[cols]), columns=cols).add_prefix('ordinal_enc_').astype('category')
    _test = pd.DataFrame(encoder.transform(test[cols]), columns=cols).add_prefix('ordinal_enc_').astype('category')
    return _train, _test


def target_encoder(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    cols: list[str], 
    target_col: str, 
    methods:list[str]=['mean', 'std']
    ):
    """
    usage:
    train_data, valid_data = target_encoder(train, valid, cat_cols, target_col)
    _, test = target_encoder(train, test, cat_cols, target_col)
    """
    train = train_df.copy()
    test = test_df.copy()
    # for test
    for col in cols:
        for method in methods:
            if method == 'mean':
                group = train.groupby(col)[target_col].mean().to_dict()
            elif method == 'std':
                group = train.groupby(col)[target_col].std().to_dict()
            elif method == 'max':
                group = train.groupby(col)[target_col].max().to_dict()
            elif method == 'min':
                group = train.groupby(col)[target_col].min().to_dict()
            elif method == 'median':
                group = train.groupby(col)[target_col].median().to_dict()
            else:
                raise ValueError(f'{method} is not implemented')
            test[f'{col}_target_enc_{method}'] = test[col].map(group)

    # for train
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(train))
    for (idx_train, idx_valid) in cv:
        for col in cols:
            for method in methods:
                if method == 'mean':
                    group = train.loc[idx_train, :].groupby(col)[target_col].mean().to_dict()
                elif method == 'std':
                    group = train.loc[idx_train, :].groupby(col)[target_col].std().to_dict()
                elif method == 'max':
                    group = train.loc[idx_train, :].groupby(col)[target_col].max().to_dict()
                elif method == 'min':
                    group = train.loc[idx_train, :].groupby(col)[target_col].min().to_dict()
                elif method == 'median':
                    group = train.loc[idx_train, :].groupby(col)[target_col].median().to_dict()
                else:
                    raise ValueError(f'{method} is not implemented')
        
                train.loc[idx_valid, f'{col}_target_enc_{method}'] = train.loc[idx_valid, col].map(group)
    output_cols = [f'{col}_target_enc_{method}' for col in cols for method in methods]
    return train[output_cols], test[output_cols]


def multilabel2onehot(df: pd.DataFrame, multilabel_cols: list[str]):
    """
    カンマ区切りのデータをonehot化
    数が多くなる場合はsvdなどで次元圧縮する
    """
    multilabel_dfs = []
    for c in multilabel_cols:
        list_srs = df[c].fillna('').map(lambda x: x.split(",")).tolist()
        mlb = MultiLabelBinarizer()
        ohe_srs = mlb.fit_transform(list_srs)
        col_df = pd.DataFrame(ohe_srs, columns=[f"ohe_{c}_{name}" for name in mlb.classes_])
        multilabel_dfs.append(col_df)

    multilabel_df = pd.concat(multilabel_dfs, axis=1) # .astype('category')
    return multilabel_df