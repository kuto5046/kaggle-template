import pandas as pd 
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def count_encoder(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """count encoding
    Args:
        df: カテゴリ変換する対象のデータフレーム
        cols (list of str): カテゴリ変換する対象のカラムリスト
    Returns:
        pd.Dataframe: dfにカテゴリ変換したカラムを追加したデータフレーム
    """
    out_df = pd.DataFrame()
    for c in cols:
        series = df[c]
        vc = series.value_counts(dropna=False)
        _df = pd.DataFrame(df[c].map(vc))
        out_df = pd.concat([out_df, _df], axis=1)

    out_df = out_df.add_suffix('_count_enc')
    return pd.concat([df, out_df], axis=1)


def ordinal_encoder(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    ce_oe = OrdinalEncoder()
    temp_df = pd.DataFrame()
    temp_df[cols] = ce_oe.fit_transform(df[cols])
    temp_df = temp_df.add_suffix('_ordinal_enc')
    df = pd.concat([df, temp_df], axis=1)
    return df