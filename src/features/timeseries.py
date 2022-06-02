import pandas as pd 
from typing import List 

def get_lag_features(df:pd.DataFrame, col: str, lag: int, group_cols:List[str]=None):
    """lag特徴量を作成する関数
    group_colsを指定するとそのgroup内部でlagをとる
    colはlagを取りたいcolumn名
    
    usage:
    df = get_lag_features(df, col='time_col', lag=1, group_cols=None)
    """
    if group_cols is None:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    else:
        group_str = "_".join(map(str, group_cols))
        df[f'{col}_lag{lag}_by_{group_str}'] = df.groupby(group_cols)[col].transform(lambda x: x.shift(lag))
    return df 


def get_moving_average_features(df:pd.DataFrame, col, lag:int, window_size: int, group_cols:List[str]=None):
    """移動平均を取る
    group_colsを指定するとそのgroup内部でlagをとる
    colはlagを取りたいcolumn名
    lagとwindow_sizeの関係: lag期間前からwindow_size分を集約する

    usage:
    df = get_lag_features(df, col='time_col', lag=1, window_size=3, group_cols=['group1', 'group2'])
    """
    if group_cols is None:
        df[f'{col}_rolling_mean_{lag}_{window_size}'] = df[col].shift(lag).rolling(window_size).mean()
        df[f'{col}_rolling_std_{lag}_{window_size}']  = df[col].shift(lag).rolling(window_size).std()
        df[f'{col}_rolling_max_{lag}_{window_size}'] = df[col].shift(lag).rolling(window_size).max()
        df[f'{col}_rolling_min_{lag}_{window_size}']  = df[col].shift(lag).rolling(window_size).min()
    else:
        group_str = "_".join(map(str, group_cols))
        df[f'{col}_rolling_mean_{lag}_{window_size}_by_{group_str}'] = df.groupby(group_cols)[col].transform(lambda x: x.shift(lag).rolling(window_size).mean())
        df[f'{col}_rolling_std_{lag}_{window_size}_by_{group_str}']  = df.groupby(group_cols)[col].transform(lambda x: x.shift(lag).rolling(window_size).std())
        df[f'{col}_rolling_max_{lag}_{window_size}_by_{group_str}'] = df.groupby(group_cols)[col].transform(lambda x: x.shift(lag).rolling(window_size).max())
        df[f'{col}_rolling_min_{lag}_{window_size}_by_{group_str}']  = df.groupby(group_cols)[col].transform(lambda x: x.shift(lag).rolling(window_size).min())
    
    return df 