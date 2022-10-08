import os 
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
from typing import Union 

tqdm.pandas()


def count_lda_vectorize(input_df:pd.DataFrame, col:str, n_components:int = 50):
    """
    usage:
    df2 = count_lda_vectorize(df, col='abc', n_components=5)
    """
    pipeline = Pipeline(steps=[
        ("CountVectorizer", CountVectorizer()),
        ("LDA", LatentDirichletAllocation(n_components=n_components, random_state=42))
    ])
    features = pipeline.fit_transform(input_df[col].fillna(""))
    output_df = pd.DataFrame(features).add_prefix(f'count_lda_{col}_')
    return output_df


def tfidf_svd_vectorize(input_df:pd.DataFrame, col:str, n_components:int = 50):
    """
    usage:
    tfidf_df = tfidf_svd_vectorize(df, col='abc', n_components=5)
    次元圧縮したくない場合はpipelineのTruncatedSVDをコメントアウト
    """
    pipeline = Pipeline(steps=[
        ("TfidfVectorizer", TfidfVectorizer()),
        ("TruncatedSVD", TruncatedSVD(n_components=n_components, random_state=42))
    ])
    features = pipeline.fit_transform(input_df[col].fillna(""))
    output_df = pd.DataFrame(features).add_prefix(f'tfidf_svd_{col}_')
    return output_df


def get_embedding_model(embedding_source='wikipedia-160', output_dir="/home/user/work/input/resource"):
    """ 
    以下のリンク先にあるembedding fileを取得し保存
    https://github.com/clips/dutchembeddings 
    """
    import urllib
    import tarfile
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{embedding_source}.tar.gz'

    if not os.path.exists(save_path):
        url = f'https://www.clips.uantwerpen.be/dutchembeddings/{embedding_source}.tar.gz'
        urllib.request.urlretrieve(url, save_path)
    
        with tarfile.open(save_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    else:
        print(f'skipped because {save_path} is already exist')



class Sentence2Vec():
    """ word2vecで単語をベクトル化し文章全体で平均を取る
    https://github.com/clips/dutchembeddings
    からモデルを取ってくる

    usage:
    model_file = 'wikipedia-160.txt'
    ndim = 160
    encoder = Sentence2Vec(model_file)
    df2 = encoder.vectorize_to_df(df, col, ndim=160)
    """
    def __init__(self, model_file="./160/wikipedia-160.txt"):
        self.w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    def vectorize(self, x: str, ndim=160):
        embeddings = [
            self.w2v_model.get_vector(word)
            if self.w2v_model.key_to_index.get(word, None) is not None
            else np.zeros(ndim, dtype=np.float32)
            for word in x.split()
        ]
        if len(embeddings) == 0:
            return np.zeros(ndim, dtype=np.float32)
        else:
            return np.mean(embeddings, axis=0)

    def vectorize_to_df(self,input_df, col, ndim=160):
        vector = np.stack(
            input_df[col].fillna("").str.replace("\n", "").progress_apply(lambda x: self.vectorize(x, ndim)).to_numpy()
            )
        output_df = pd.DataFrame(vector).add_prefix('senentce2vec_')
        return output_df 


class SCDVEmbedder(TransformerMixin, BaseEstimator):
    """
    usage:
    model_file="./320/wikipedia-320.txt"
    w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    model = SCDVEmbedder(w2v_model, k=5)
    model.fit(df["title"])
    features = model.transform(df["title"])
    """
    def __init__(self, w2v, k=5):
        self.w2v = w2v
        self.vocab = set(self.w2v.vocab.keys())
        self.tokenizer = self.tokenizer
        self.k = k
        self.topic_vector = None
        self.tv = TfidfVectorizer(tokenizer=self.tokenizer)

    def tokenizer(x: str):
        return x.split()

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


class UniversalSentenceEncoder():
    """
    usage:
    usencoder = UniversalSentenceEncoder()  # 必要に応じてloadするモデルを変更
    s1_vec = usencoder.vectorize(whole, col='s1')
    s2_vec = usencoder.vectorize(whole, col='s2')
    s1_s2_sim = usencoder.similarity_vectorize(s1_vec, s2_vec)
    """
    def __init__(self):
        self.embedder = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )

    def vectorize(self, input_df: pd.DataFrame, col: str, save: bool=True) -> pd.DataFrame:
        vector = np.stack(
            input_df[col].fillna("").progress_apply(
                lambda x: self.embedder(x).numpy().reshape(-1)
                ).values
        )
        output_df = pd.DataFrame(vector).add_prefix(f'universal_sentence_{col}')
        if save:
            output_df.to_pickle(f'universal_sentence_{col}.pkl')
        return output_df

    def similarity_vectorize(
        self, 
        vec1: Union[pd.DataFrame, np.ndarray], 
        vec2: Union[pd.DataFrame, np.ndarray],
        save: bool=True) -> pd.DataFrame:

        if isinstance(vec1, np.ndarray) & isinstance(vec1, np.ndarray):
            pass
        elif isinstance(vec1, pd.DataFrame) & isinstance(vec1, pd.DataFrame):
            vec1 = vec1.to_numpy()
            vec2 = vec2.to_numpy()
        else:
            TypeError
        similarity = (vec1 * vec2).sum(axis=1) / (
            np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
        output_df = pd.DataFrame()
        output_df['universal_sentence_similarity'] = similarity
        if save:
            output_df.to_pickle(f'sentence_vec_similarity.pkl')
        return output_df


class BertSequenceVectorizer:
    def __init__(self, model_name="bert-base-uncased", max_len=128):
        """
        usage:
        bert = BertSequenceVectorizer(
            model_name="bert-base-multilingual-uncased",
            max_len=128
            )
        df = bert.vectorize_to_df(df, col)
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

    def vectorize_to_df(self, input_df: pd.DataFrame, col:str):
        vector = np.stack(
            input_df[col].fillna("").progress_apply(
                lambda x: self.vectorize(x).reshape(-1)).to_numpy()
            )
        output_df = pd.DataFrame(vector).add_prefix(f'bert_{col}_')
        return output_df 
