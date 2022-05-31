import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder 
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


def count_encoder(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    usage:
    train, test = count_encoder(train, test, cols=['cat1', 'cat2'])
    """
    for col in cols:
        encoder = train[col].value_counts()
        train[f'{col}_count_enc'] = train[col].map(encoder)
        test[f'{col}_count_enc'] = test[col].map(encoder)
    return train, test


def label_encoder(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    ordinal_encoderと同じ挙動のはず
    usage:
    train, test = label_encoder(train, test, cols=['cat1', 'cat2'])
    """
    for col in cols:
        encoder = LabelEncoder()
        _whole = pd.concat([train, test], axis=0).reset_index(drop=True)
        encoder.fit(_whole[col])
        train[f'{col}_label_enc'] = encoder.transform(train[col])
        test[f'{col}_label_enc'] = encoder.transform(test[col])
    return train, test


def ordinal_encoder(train: pd.DataFrame, test: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    label_encoderと同じ挙動のはず
    usage:
    train, test = ordinal_encoder(train, test, cols=['cat1', 'cat2'])
    """
    encoder = OrdinalEncoder()
    _whole = pd.concat([train, test], axis=0).reset_index(drop=True)
    encoder.fit(_whole[cols])
    _train = pd.DataFrame(encoder.transform(train[cols]), columns=cols).add_suffix('_ordinal_enc')
    _test = pd.DataFrame(encoder.transform(test[cols]), columns=cols).add_suffix('_ordinal_enc')
    train = pd.concat([train, _train], axis=1)
    test = pd.concat([test, _test], axis=1)
    return train, test



def target_encoder(train: pd.DataFrame, test: pd.DataFrame, col: str, target_col: str):
    """
    usage:
    
    train_data = train.loc[idx_train].reset_index(drop=True)
    valid_data = train.loc[idx_valid].reset_index(drop=True)
    for col in cat_cols:
        train_data, valid_data = target_encoder(train_data, valid_data, col, target_col)
        _, test = target_encoder(train, test, col, target_col)
    feature_cols = train_data.drop(drop_cols, axis=1).columns
    """
    # for test
    group = train.groupby(col)[target_col].mean().to_dict()
    test[f'{col}_te'] = test[col].map(group)

    # for train
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(train))
    for (idx_train, idx_valid) in cv:
        group = train.loc[idx_train, :].groupby(col)[target_col].mean().to_dict()
        train.loc[idx_valid, f'{col}_te'] = train.loc[idx_valid, col].map(group)
    return train, test