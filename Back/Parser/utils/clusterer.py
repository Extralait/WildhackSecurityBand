import os
import pickle
import re
import traceback

import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from Back.Parser.config import DATA_DIR


class Clusterer:

    def __init__(self, file_names_prefix="", load_valid_data=True, load_valid_popularity_data=False,
                 load_mini_batch_k_means=False, load_tfidf_vectorizer=False):
        self.morph = MorphAnalyzer()
        self.file_names_prefix = file_names_prefix
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, max_df=0.6, min_df=0.01, max_features=100000,
                                                use_idf=True,stop_words=stopwords.words('russian'), tokenizer=self.token_only, ngram_range=(1, 3))

        if load_valid_data:
            try:
                self.load_valid_data()
                print('Данные загружены')
            except:
                print('Невозможно загрузить данные')
                traceback.print_exc()
                self.tokenizer = None

        if load_valid_popularity_data:
            try:
                self.load_valid_popularity_data()
                print('Данные загружены')
            except:
                print('Невозможно загрузить данные')
                traceback.print_exc()
                self.valid_popularity_data = None

        if load_mini_batch_k_means:
            try:
                self.load_mini_batch_k_means()
                print('Минибачер загружен')
            except:
                print('Невозможно загрузить минибачер')
                traceback.print_exc()
                self.mini_batch_k_means = None

        if load_tfidf_vectorizer:
            try:
                self.load_tfidf_vectorizer()
                print('Векторизатор загружен')
            except:
                print('Невозможно обучить модель')
                traceback.print_exc()

    def load_valid_data(self):
        self.valid_data = pd.read_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}valid_data.csv"))

    def load_valid_popularity_data(self):
        self.valid_popularity_data = pd.read_csv(
            os.path.join(DATA_DIR, f"{self.file_names_prefix}valid_popularity_data.csv"))

    def load_mini_batch_k_means(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}mini_batch_k_means.pickle"), 'rb') as handle:
            self.mini_batch_k_means = pickle.load(handle)

    def load_tfidf_vectorizer(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}tf_idf_matrix.pickle"), 'rb') as handle:
            self.tfidf_vectorizer = pickle.load(handle)

    def save_clusterer(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}mini_batch_k_means.pickle"), 'wb') as handle:
            pickle.dump(self.mini_batch_k_means, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}tf_idf_matrix.pickle"), 'wb') as handle:
            pickle.dump(self.tfidf_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def token_only(self, text):
        bad_words = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_{|}~\"\-]+"
        text = re.sub(bad_words, ' ', text)
        tokens = [word.lower() for sent in sent_tokenize(text) for word in word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            token = self.morph.normal_forms(token)[0]
            filtered_tokens.append(token)
        return filtered_tokens

    def add_data(self, data, tf_idf_matrix):
        y_kmeansMBK = self.mini_batch_k_means.predict(tf_idf_matrix)

        Num = [pt for pt in y_kmeansMBK]
        df2 = {"Num_Cluster": Num}
        dfMBK = pd.DataFrame(df2)
        df = pd.concat([data, dfMBK], axis=1)
        return df

    def train_clusterer(self):
        train_data = self.valid_data['UQ']

        self.tf_idf_matrix = self.tfidf_vectorizer.fit_transform(train_data.values.astype('U'))

        self.mini_batch_k_means = MiniBatchKMeans(n_clusters=3000, init='random').fit(self.tf_idf_matrix)

        self.save_clusterer()

        df = self.add_data(self.valid_data, self.tf_idf_matrix)

        df.to_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}claster.csv"), index=False)

    def comparsion_clustering(self):
        data = self.valid_popularity_data['query']

        tf_idf_matrix = self.tfidf_vectorizer.transform(data.values.astype('U'))

        df = self.add_data(self.valid_popularity_data,tf_idf_matrix)

        df.to_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}clastered_popularity_data.csv"), index=False)

    def predict(self, text):
        tf_idf_matrix = self.tfidf_vectorizer.transform([text])
        y_kmeansMBK = self.mini_batch_k_means.predict(tf_idf_matrix)
        return y_kmeansMBK[0]

