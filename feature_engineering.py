import sys 
import pandas as pd
from src import utils as u
from src.cv import get_fold
from src.features.base import Feature, generate_features, get_categorical_col, get_numerical_col
from src.features.encoder import (count_encoder, ordinal_encoder,
                                  pp_for_categorical_encoding)
from src.features.nlp import UniversalSentenceEncoder

class CountEncode(Feature):
    def create_features(self):
        _train, _test = count_encoder(train, test, ['file_extension'])
        self.train = _train.filter(like='count_enc')
        self.test = _test.filter(like='count_enc')


class OrdinalEncode(Feature):
    def create_features(self):
        _train, _test = ordinal_encoder(train, test, cat_cols)
        self.train = _train.filter(like='ordinal_enc').astype('category')
        self.test = _test.filter(like='ordinal_enc').astype('category')

# class Numerical(Feature):
#     def create_features(self):
#         cols = get_numerical_col(train, skip_cols=[target_col])
#         self.train = train[cols]
#         self.test = test[cols]


class StringLength(Feature):
    def create_features(self):
        for col in train.filter(like='code'):
            self.train[f'string_length_{col}'] = train[col].str.len()
            self.test[f'string_length_{col}'] = test[col].str.len()

class CountSign(Feature):
    def create_features(self):
        for col in train.filter(like='code'):
            for i, sign in enumerate(["\'", "\"", "\?", "!", ",", "."]):
                self.train[f'count_{col}_{i}'] = train[col].str.count(sign)
                self.test[f'count_{col}_{i}'] = test[col].str.count(sign)


class USEncode(Feature):
    def create_features(self):
        usencoder = UniversalSentenceEncoder()
        self.train = usencoder.vectorize(train, col='code_2')
        self.test = usencoder.vectorize(test, col='code_2')

class IsAssert(Feature):
    def create_features(self):
        self.train['is_contain_assert'] = ((train['code_2'].str.contains('assert'))|(train['code_2'].str.contains('ASSERT')))*1
        self.test['is_contain_assert'] = ((test['code_2'].str.contains('assert'))|(test['code_2'].str.contains('ASSERT')))*1
        self.train['is_contain_is'] = ((train['code_2'].str.contains('is'))|(train['code_2'].str.contains('IS')))*1
        self.test['is_contain_is'] = ((test['code_2'].str.contains('is'))|(test['code_2'].str.contains('IS')))*1

class IsException(Feature):
    def create_features(self):
        self.train['is_contain_raise'] = (train['code_2'].str.contains('raise'))*1
        self.test['is_contain_raise'] = (test['code_2'].str.contains('raise'))*1
        self.train['is_contain_exception'] = (train['code_2'].str.contains('exception'))*1
        self.test['is_contain_exception'] = (test['code_2'].str.contains('exception'))*1


if __name__ == '__main__':
    config = u.Config().get_cnf('../../configs/', 'config.yaml')
    INPUT_DIR = config.common.input_dir
    target_col = config.common.target_col

    train = pd.read_csv(INPUT_DIR + 'train.csv')
    test = pd.read_csv(INPUT_DIR + 'test.csv')

    cat_cols = get_categorical_col(train, skip_cols=['id'])
    train, test = pp_for_categorical_encoding(train, test, cat_cols)

    generate_features(globals())
