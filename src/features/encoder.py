import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder 


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