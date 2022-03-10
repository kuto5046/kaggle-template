import pandas as pd

def get_category_col(df:pd.DataFrame):
    """カテゴリ型のカラム名を取得"""
    category_cols = []
    numerics = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type not in numerics:
            category_cols.append(col)
    return category_cols


def get_num_col(df:pd.DataFrame):
    """数値型のカラム名を取得"""
    num_cols = []
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            num_cols.append(col)
    return num_cols