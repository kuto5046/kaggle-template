import pandas as pd 
import numpy as np 
from typing import List 
from datetime import timedelta

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


def create_datetime_feature(
    df: pd.DataFrame, 
    col: str, 
    prefix: str = None, 
    change_utc2asia: bool = False,
    attrs: list = ['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']) -> pd.DataFrame:
    """日時特徴量の生成処理
    Args:
        df (pd.DataFrame): 日時特徴量を含むDF
        col (str)): 日時特徴量のカラム名
        prefix (str): 新しく生成するカラム名に付与するprefix
        attrs (list of str): 生成する日付特徴量. Defaults to ['year', 'quarter', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute']
                             cf. https://qiita.com/Takemura-T/items/79b16313e45576bb6492
    Returns:
        pd.DataFrame: 日時特徴量を付与したDF

    usage:
    df = create_datatime_feature(df, col='timestamp', attrs=['year', 'month', 'week', 'day', 'dayofweek', 'hour', 'minute'])
    """
    if prefix is None:
        prefix = col

    # utc -> asia/tokyo
    if change_utc2asia:
        df.loc[:, col] = pd.to_datetime(df[col]) + timedelta(hours=9)
    else:
        df.loc[:, col] = pd.to_datetime(df[col])

    for attr in attrs:
        dtype = np.int16 if attr == 'year' else np.int8
        df[prefix + '_' + attr] = getattr(df[col].dt, attr).astype(dtype)

    # 土日フラグ
    if 'dayofweek' in attrs:
        df[prefix + '_is_weekend'] = df[prefix + '_dayofweek'].isin([5, 6]).astype(np.int8)

    # 時間帯情報
    if 'hour' in attrs:
        df[prefix + '_hour_zone'] = pd.cut(df[prefix + '_' + 'hour'].values, bins=[-np.inf, 6, 12, 18, np.inf]).codes

    # 日付の周期性を算出
    def sin_cos_encode(df, col):
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
        return df

    for col in [prefix + '_' + 'quarter', prefix + '_' + 'month', prefix + '_' + 'day', prefix + '_' + 'dayofweek',
                prefix + '_' + 'hour', prefix + '_' + 'minute', prefix + '_' + 'hour_zone']:
        
        if col in df.columns.tolist():
            df = sin_cos_encode(df, col)

    return df