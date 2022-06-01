import hydra
import sys 
import pandas as pd
from omegaconf import DictConfig
sys.path.append("../../")
from src import utils as u
from src.cv import get_fold
from src.features.base import Feature, generate_features, get_categorical_col, get_numerical_col
from src.features.encoder import (count_encoder, ordinal_encoder,
                                  pp_for_categorical_encoding)


class CountEncode(Feature):
    def create_features(self):
        _train, _test = count_encoder(train, test, cat_cols)
        self.train = _train.filter(like='count_enc')
        self.test = _test.filter(like='count_enc')


class OrdinalEncode(Feature):
    def create_features(self):
        _train, _test = ordinal_encoder(train, test, cat_cols)
        self.train = _train.filter(like='ordinal_enc').astype('category')
        self.test = _test.filter(like='ordinal_enc').astype('category')

class Numerical(Feature):
    def create_features(self):
        cols = get_numerical_col(train, skip_cols=['id', target_col])
        self.train = train[cols]
        self.test = test[cols]

if __name__ == '__main__':
    config = u.Config().get_cnf('../../configs/', 'default.yaml')
    INPUT_DIR = config.common.input_dir
    target_col = config.common.target_col

    train = pd.read_csv(INPUT_DIR + 'train_ratings.csv')
    test = pd.read_csv(INPUT_DIR + 'test_ratings.csv')
    users = pd.read_csv(INPUT_DIR + 'users.csv')
    items = pd.read_csv(INPUT_DIR + 'books.csv')
    train = train.merge(items, on='book_id', how='left').merge(users, on='user_id', how='left')
    test = test.merge(items, on='book_id', how='left').merge(users, on='user_id', how='left')

    # pp
    train['year'].replace({'DK Publishing Inc': 0, '0': 0}, inplace=True)
    test['year'].replace({'0': 0}, inplace=True)
    train['year'] = train['year'].astype(int)
    test['year'] = test['year'].astype(int)

    cat_cols = get_categorical_col(train, skip_cols=['id'])
    train, test = pp_for_categorical_encoding(train, test, cat_cols)

    generate_features(globals())
