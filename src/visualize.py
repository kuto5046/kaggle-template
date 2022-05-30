import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from typing import List 
import lightgbm as lgb 
import numpy as np
from matplotlib_venn import venn2
import sweetviz as sv
import pandas_profiling as pdp

def pd_profiling(df: pd.DataFrame, output_dir: str='./'):
    """
    usage:
    pd_profiling(whole)
    """
    profile = pdp.ProfileReport(df)
    profile.to_file(output_file=output_dir + "profile.html")


def sweetviz_report(
    train:pd.DataFrame, 
    test:pd.DataFrame, 
    target_col:str=None, 
    skip_cols:List[str]=[], 
    output_dir:str='./'):
    """
    usage:
    sweetviz_report(train, test, target_col='target')
    """

    feature_config = sv.FeatureConfig(skip=skip_cols)

    my_report = sv.compare(
        [train, "Training Data"], 
        [test, "Test Data"], 
        target_col, 
        feature_config)
    my_report.show_html(output_dir + "SWEETVIZ_REPORT.html")


def get_uniques(input_df: pd.DataFrame, column):
    s = input_df[column]
    return set(s.dropna().unique())

def plot_intersection(
    left: pd.DataFrame, 
    right: pd.DataFrame, 
    target_column: str, 
    ax: plt.Axes = None, 
    set_labels: List[str]=None
    ):

    venn2(
        subsets=(get_uniques(left, target_column), get_uniques(right, target_column)),
        set_labels=set_labels or ("Train", "Test"),
        ax=ax
        )
    ax.set_title(target_column)

def plot_venn(train:pd.DataFrame, test:pd.DataFrame, output_dir: str='./'):
    """
    usage:
    plot_venn(train, test)
    """
    target_columns = test.columns.tolist()
    n_cols = 5
    n_rows = - (- len(target_columns) // n_cols)

    fig, axes = plt.subplots(figsize=(4 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows)
    for c, ax in zip(target_columns, np.ravel(axes)):
        plot_intersection(train, test, target_column=c, ax=ax)
    plt.savefig(output_dir + 'venn.png')


def plot_importance(
    models:List[lgb.Booster], 
    feat_train_df:pd.DataFrame, 
    output_dir: str='./',
    top_n:int =50,
    ):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    usage:
    ```
    fig, ax = visualize_importance(models, train_feat_df)
    ```
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importance()
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:top_n]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x='feature_importance', 
                  y='column', 
                  order=order, 
                  ax=ax, 
                  palette='viridis', 
                  orient='h')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Importance')
    ax.grid()
    fig.tight_layout()
    plt.savefig(output_dir + 'importance.png')
    return fig, ax