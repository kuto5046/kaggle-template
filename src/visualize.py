import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from typing import List 
import lightgbm as lgb 
import numpy as np
from matplotlib_venn import venn2
import sweetviz as sv
from IPython.display import display_html
import xgboost
import lightgbm
import catboost
from pathlib import Path

# 欠損値の確認
def missing_values(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# 頻出値の確認
def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))


# ユニーク値の確認
def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    tt['Percent'] = np.round(uniques / total * 100, 3)
    return(np.transpose(tt))


# これを呼ぶ
def show_all(df, n=5):
    print(f'Data shape :{df.shape}')
    df_describe = df.describe().style.set_table_attributes("style='display:inline'").set_caption('Describe Data Num')
    df_describe_o = df.describe(include=[object]).style.set_table_attributes("style='display:inline'").set_caption('Describe Data Object')
    df_head = df.head(n).style.set_table_attributes("style='display:inline'").set_caption('Head Data')
    df_tail = df.tail(n).style.set_table_attributes("style='display:inline'").set_caption('Tail Data')
    df_missing = missing_values(df).style.set_table_attributes("style='display:inline'").set_caption('Missing Value')
    df_frequent = most_frequent_values(df).style.set_table_attributes("style='display:inline'").set_caption('Frequent Value')
    df_unique = unique_values(df).style.set_table_attributes("style='display:inline'").set_caption('Unique Value')

    display_html(df_describe._repr_html_(), raw=True)
    display_html(df_describe_o._repr_html_(), raw=True)
    display_html(df_head._repr_html_(), raw=True)
    display_html(df_tail._repr_html_(), raw=True)
    display_html(df_missing._repr_html_(), raw=True)
    display_html(df_frequent._repr_html_(), raw=True)
    display_html(df_unique._repr_html_(), raw=True)
    return None


def pd_profiling(df: pd.DataFrame, output_dir: Path = Path('./')):
    """
    usage:
    pd_profiling(whole)
    """
    import ydata_profiling
    profile_report = ydata_profiling.ProfileReport(df)
    profile_report.to_file(output_dir / "report.html")


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
    models:List[lgb.Booster] | List[xgboost.core.Booster] | List[catboost.core.CatBoostRegressor] | List[catboost.core.CatBoostClassifier], 
    output_dir: Path=Path('./'),
    importance_type='gain',
    top_n:int =50,
    ):
    """model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    args:
        models:
            List of lightGBM models
        feature_cols:
            学習時に使った特徴量リスト
    usage:
    ```
    fig, ax = visualize_importance(models, train_feat_df)
    ```
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()

        if isinstance(model, lightgbm.Booster):
            _df['feature_importance'] = model.feature_importance(importance_type=importance_type)
            _df['column'] = model.feature_name()
        elif isinstance(model, xgboost.core.Booster):
            feature_importance = model.get_score(importance_type=importance_type)
            _df['feature_importance'] = feature_importance.values()
            _df['column'] = feature_importance.keys()
        elif isinstance(model, catboost.core.CatBoostRegressor) or isinstance(model, catboost.core.CatBoostClassifier):
            _df['feature_importance'] = model.get_feature_importance()
            _df['column'] = model.feature_names_
        else:
            raise NotImplementedError
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
    plt.savefig(output_dir / 'importance.png')
    return fig, ax


def plot_shap(model, df, feature_cols, plot_type=None):
    import shap 
    shap.initjs()
    explainer = shap.TreeExplainer(model, data=df[feature_cols], check_additivity=False)
    shap_values = explainer.shap_values(df[feature_cols])
    shap.summary_plot(shap_values=shap_values,
                    features=df[feature_cols],
                    feature_name=feature_cols,
                    plot_type=plot_type,
                    )