import os
import pickle
import traceback

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

from Back.Parser.config import DATA_DIR
from Back.Parser.utils.limiter import limiter

class UQTokenizer:

    def __init__(self,file_names_prefix="", load_data=True, load_tokenizer=False):
        self.file_names_prefix = file_names_prefix
        self.tokenizer = None
        if load_data:
            try:
                self.load_data()
                print('Данные  загружены')
            except:
                print('Невозможно загрузить данные')
                traceback.print_exc()
                self.data = None

        if load_tokenizer:
            try:
                self.load_tokenizer()
                print('Токенайзер загружен')
            except:
                print('Невозможно загрузить токенайзер')
                traceback.print_exc()
                self.tokenizer = None

    def load_data(self):
        self.data = pd.read_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}claster.csv"))

    def load_tokenizer(self):
        with open(os.path.join(DATA_DIR, "tokenizer.pickle"), 'rb') as handle:
            self.tokenizer = pickle.load(handle)


    def train_tokenizer(self):
        text = limiter(self.data.UQ.to_list())

        df2 = {"limited_UQ": list(map(lambda x:' '.join(x),text))}
        dfMBK = pd.DataFrame(df2)
        df = pd.concat([self.data,dfMBK], axis=1)

        df.to_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}train_data.csv"), index=False)

        self.tokenizer = Tokenizer(num_words=10000)

        self.tokenizer.fit_on_texts(text)

        self.save_tokenizer()


    def save_tokenizer(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}tokenizer.pickle"), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

