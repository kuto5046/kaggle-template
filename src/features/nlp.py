from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
import fasttext
import fasttext.util
from gensim.models import KeyedVectors
import torch
import transformers
from transformers import BertTokenizer
import numpy as np
import pandas as pd 
from typing import List 
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from tqdm import tqdm
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.mixture import GaussianMixture


tqdm.pandas()


def count_vectorize(input_df:pd.DataFrame, cols:List[str]):
    """
    文章中のtokenの頻度を数えたスパースマトリクスを作成。
    行列の各行が各文章に該当し、各列がtokenに対応。
    つまり文章をあるtokenがあるかないかで特徴づけ、ベクトルを得る手法
    """
    output_list = []
    for col in cols:
        cv = CountVectorizer()
        features = cv.fit_transform(input_df[col].fillna(""))
        out_df = pd.DataFrame(features).add_prefix(f"{col}_tfidf_")
        output_list.append(out_df)
    output_df = pd.concat(output_list)
    return output_df


def tfidf_svd_vectorize(input_df:pd.DataFrame, cols:List[str], n_components:int =50):
    output_df = input_df.copy()
    for col in cols:
        tfidf_svd = Pipeline(steps=[
            ("TfidfVectorizer", TfidfVectorizer()),
            ("TruncatedSVD", TruncatedSVD(n_components=n_components, random_state=42))
        ])
        features_svd = tfidf_svd.fit_transform(input_df["title"].fillna(""))


def get_sentence_vector(x: str, ndim=320):
    """
    usage:
    features = np.stack(
        df["title"].fillna("").str.replace("\n", "").map(lambda x: get_sentence_vector(x)
        ).values
    """
    embeddings = [
        w2v_model.get_vector(word)
        if w2v_model.vocab.get(word) is not None
        else np.zeros(ndim, dtype=np.float32)
        for word in x.split()
    ]
    if len(embeddings) == 0:
        return np.zeros(ndim, dtype=np.float32)
    else:
        return np.mean(embeddings, axis=0)




class SCDVEmbedder(TransformerMixin, BaseEstimator):
    """
    usage:
    scdv = SCDVEmbedder(w2v_model, tokenizer=tokenizer, k=5)
    scdv.fit(df["title"])
    features = scdv.transform(df["title"])
    """
    def __init__(self, w2v, tokenizer, k=5):
        self.w2v = w2v
        self.vocab = set(self.w2v.vocab.keys())
        self.tokenizer = tokenizer
        self.k = k
        self.topic_vector = None
        self.tv = TfidfVectorizer(tokenizer=self.tokenizer)

    def __assert_if_not_fitted(self):
        assert self.topic_vector is not None, \
            "SCDV model has not been fitted"

    def __create_topic_vector(self, corpus: pd.Series):
        self.tv.fit(corpus)
        self.doc_vocab = set(self.tv.vocabulary_.keys())

        self.use_words = list(self.vocab & self.doc_vocab)
        self.use_word_vectors = np.array([
            self.w2v[word] for word in self.use_words])
        w2v_dim = self.use_word_vectors.shape[1]
        self.clf = GaussianMixture(
            n_components=self.k, 
            random_state=42,
            covariance_type="tied")
        self.clf.fit(self.use_word_vectors)
        word_probs = self.clf.predict_proba(
            self.use_word_vectors)
        world_cluster_vector = self.use_word_vectors[:, None, :] * word_probs[
            :, :, None]

        doc_vocab_list = list(self.tv.vocabulary_.keys())
        use_words_idx = [doc_vocab_list.index(w) for w in self.use_words]
        idf = self.tv.idf_[use_words_idx]
        topic_vector = world_cluster_vector.reshape(-1, self.k * w2v_dim) * idf[:, None]
        topic_vector = np.nan_to_num(topic_vector)

        self.topic_vector = topic_vector
        self.vocabulary_ = set(self.use_words)
        self.ndim = self.k * w2v_dim

    def fit(self, X, y=None):
        self.__create_topic_vector(X)

    def transform(self, X):
        tokenized = X.fillna("").map(lambda x: self.tokenizer(x))

        def get_sentence_vector(x: list):
            embeddings = [
                self.topic_vector[self.use_words.index(word)]
                if word in self.vocabulary_
                else np.zeros(self.ndim, dtype=np.float32)
                for word in x
            ]
            if len(embeddings) == 0:
                return np.zeros(self.ndim, dtype=np.float32)
            else:
                return np.mean(embeddings, axis=0)
        return np.stack(
            tokenized.map(lambda x: get_sentence_vector(x)).values
        )


def tokenizer(x: str):
    return x.split()


class UniversalSentenceEncoder():
    def __init__(self):
        self.embedder = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )

    def vectorize(self, input_df, col):
        features = np.stack(
            input_df[col].fillna("").progress_apply(
                lambda x: self.embedder(x).numpy().reshape(-1)
                ).values
        )
        tmp_df = pd.DataFrame(features).add_suffix(f'_{col}_sentence_vec')
        output_df = pd.concat([input_df, tmp_df], axis=1)
        return output_df

    def similarity_vectorize(self, input_df, col1, col2):
        features1 = np.stack(
            input_df[col1].fillna("").progress_apply(
                lambda x: self.embedder(x).numpy().reshape(-1)
                ).values
        ) 
        features2 = np.stack(
            input_df[col2].fillna("").progress_apply(
                lambda x: self.embedder(x).numpy().reshape(-1)
                ).values
        )
        value = (features1 * features2).sum(axis=1) / (
            np.linalg.norm(features1, axis=1) * np.linalg.norm(features2, axis=1))
        input_df[f'{col1}_{col2}_sentence_sim_vec'] = value
        return input_df


class BertSequenceVectorizer:
    def __init__(self, model_name="bert-base-uncased", max_len=128):
        """
        usage:
        BSV = BertSequenceVectorizer(
            model_name="bert-base-multilingual-uncased",
            max_len=128
            )
        features = np.stack(
            df["title"].fillna("").map(lambda x: BSV.vectorize(x).reshape(-1)).values
            )
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()