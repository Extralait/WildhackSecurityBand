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


class ThemeCategorizer:

    def __init__(self, file_names_prefix="", load_tokenizer=True, load_model=True, train_model=False):
        self.file_names_prefix = file_names_prefix
        if load_tokenizer:
            try:
                self.load_tokenizer()
                print('Токенизатор загружен')
            except:
                print('Невозможно загрузить токенизатор')
                traceback.print_exc()
                self.tokenizer = None

        if load_model:
            try:
                self.load_model()
                print('Модель загружена')
            except:
                print('Невозможно загрузить модель')
                traceback.print_exc()
                self.model = None

        if train_model:
            try:
                self.train_model()
            except:
                print('Невозможно обучить модель')
                traceback.print_exc()

    def load_model(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_model.json"), 'r') as f:
            loaded_model = model_from_json(f.read())

        loaded_model.load_weights(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_weights.h5"))
        self.model = loaded_model

    def load_tokenizer(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}tokenizer.pickle"), 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def save_model(self):
        model_json = self.model.to_json()

        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_model.json"), 'w') as f:
            f.write(model_json)
        print('Модель сохранена')

    def build_model(self, max_features, maxSequenceLength, num_classes):
        print(u'Собираем модель...')

        model = Sequential()
        model.add(Embedding(max_features, maxSequenceLength))
        model.add(LSTM(5, dropout=0.2, recurrent_dropout=0.2))
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
        weight_file_path = os.path.join(DATA_DIR, f"{self.file_names_prefix}categorizer_weights.h5", )
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

    def load_data_from_arrays(self, strings, labels, train_test_split=0.9):
        data_size = len(strings)
        test_size = int(data_size - round(data_size * train_test_split))
        print("Test size: {}".format(test_size))

        print("\nTraining set:")
        x_train = strings[test_size:]
        print("\t - x_train: {}".format(len(x_train)))
        y_train = labels[test_size:]
        print("\t - y_train: {}".format(len(y_train)))

        print("\nTesting set:")
        x_test = strings[:test_size]
        print("\t - x_test: {}".format(len(x_test)))
        y_test = labels[:test_size]
        print("\t - y_test: {}".format(len(y_test)))

        return x_train, y_train, x_test, y_test

    def data_choice(self):
        data = pd.read_csv(os.path.join(DATA_DIR, f"{self.file_names_prefix}train_data.csv"))
        return data['limited_UQ'], data['Num_Cluster']

    def data_preparation(self):

        strings, categories = self.data_choice()

        X_train, y_train, X_test, y_test = self.load_data_from_arrays(strings.tolist(), categories,
                                                                      train_test_split=0.8)

        max_words = 0
        for desc in strings:
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальная длина описания: {} слов'.format(max_words))

        total_unique_words = len(self.tokenizer.word_counts)
        print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

        maxSequenceLength = max_words

        num_classes = np.max(y_train) + 1
        print('Количество категорий для классификации: {}'.format(num_classes))

        print(u'Преобразуем описания заявок в векторы чисел...')

        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
        X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

        print('Размерность X_train:', X_train.shape)
        print('Размерность X_test:', X_test.shape)

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

        return total_unique_words, maxSequenceLength, num_classes, X_train, y_train, X_test, y_test

    def train_model(self):

        total_unique_words, maxSequenceLength, num_classes, X_train, y_train, X_test, y_test = self.data_preparation()

        max_features = total_unique_words

        self.build_model(max_features, maxSequenceLength, num_classes)

        batch_size = 32

        self.fit_current_model([X_train, y_train, X_test, y_test],
                               batch_size=batch_size,
                               epochs=7, )

        score = self.model.evaluate(X_test, y_test,
                                    batch_size=batch_size, verbose=1)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

    def predict(self, text):
        text = self.tokenizer.texts_to_sequences(limiter([text])[0])
        res = self.model.predict(text)
        return np.argmax(res)
