import os
import pickle
import traceback

import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import sequence

from Back.Parser.config import DATA_DIR
from Back.Parser.utils.limiter import limiter
from Back.Parser.utils.theme_categorizer import ThemeCategorizer


class CountCategorizer(ThemeCategorizer):

    def load_model(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_model_2.json"), 'r') as f:
            loaded_model = model_from_json(f.read())

        loaded_model.load_weights(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_weights_2.h5"))
        self.model = loaded_model

    def save_model(self):
        model_json = self.model.to_json()

        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_model_2.json"), 'w') as f:
            f.write(model_json)
        print('Модель сохранена')

    def build_model(self, max_features, maxSequenceLength, num_classes):
        print(u'Собираем модель...')

        model = Sequential()
        model.add(Embedding(max_features, maxSequenceLength))
        model.add(LSTM(12, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(num_classes*6, activation='relu'))
        model.add(Dense(num_classes*3, activation='relu'))
        model.add(Dense(num_classes, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())
        self.model = model
        self.save_model()

    def fit_current_model(self,
                          data: [list, list, list, list],
                          batch_size=32,
                          epochs=5):
        weight_file_path = os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_weights_2.h5", )
        callback = ModelCheckpoint(weight_file_path,
                                   mode='max',
                                   save_best_only=True)

        X_train, y_train, X_test, y_test = data

        print(u'Тренируем модель...')
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       callbacks=[callback])

    def train_model(self):

        total_unique_words, maxSequenceLength, num_classes, X_train, y_train, X_test, y_test = self.data_preparation()

        max_features = total_unique_words

        self.build_model(max_features, maxSequenceLength, num_classes)

        batch_size = 32

        self.fit_current_model([X_train, y_train, X_test, y_test],
                               batch_size=batch_size,
                               epochs=10, )

        score = self.model.evaluate(X_test, y_test,
                                    batch_size=batch_size, verbose=1)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))


    def data_choice(self):
        data = pd.read_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}train_data.csv"))
        return data['limited_UQ'], data['cnt']

