import json
import sys

from Back.Parser.utils.clusterer import Clusterer
from Back.Parser.utils.converters import convert_search_data, convert_query_popularity, convert_comparsion_data_to_json, \
    replace_quantity_to_categories, convert_train_data_to_reserv_json
from Back.Parser.utils.count_categorizer import CountCategorizer
from Back.Parser.utils.limiter import limit_comparsion_data
from Back.Parser.utils.searcher import Searcher
from Back.Parser.utils.theme_categorizer import ThemeCategorizer
from Back.Parser.utils.tokenizer import UQTokenizer
import time

def full_search_train(file_names_prefix, nrows):
    print('Start read search history date')
    convert_search_data(file_names_prefix, nrows)
    print('End read search history date')

    print('Start clustering')
    Clusterer(file_names_prefix).train_clusterer()
    print('End clustering')

    print('Start tokenizing')
    UQTokenizer(file_names_prefix).train_tokenizer()
    print('End tokenizing')

    print('Start replacing')
    replace_quantity_to_categories(file_names_prefix)
    print('End replacing')

    print('Start categorizing training')
    ThemeCategorizer(file_names_prefix, load_model=False, train_model=True)
    CountCategorizer(file_names_prefix, load_model=False, train_model=True)


def prepare_comparsion_data(file_names_prefix):
    print('Start read query popularity')
    convert_query_popularity(file_names_prefix)
    print('End read query popularity')

    print('Start clustering')
    Clusterer(file_names_prefix,
              load_valid_data=False,
              load_valid_popularity_data=True,
              load_mini_batch_k_means=True,
              load_tfidf_vectorizer=True).comparsion_clustering()
    print('End clustering')

    print('Start limitation')
    limit_comparsion_data(file_names_prefix)
    print('End limitation')

    print('Start convert in JSON')
    convert_comparsion_data_to_json(file_names_prefix)
    print('End convert in JSON')

    print('Start convert in JSON')
    convert_train_data_to_reserv_json(file_names_prefix)
    print('End convert in JSON')


# full_search_train('example_',1000000)
# prepare_comparsion_data('example_')


model = ThemeCategorizer('example_')
searcher = Searcher('example_')
clasterer = Clusterer('example_',
                      load_valid_data=False,
                      load_valid_popularity_data=False,
                      load_mini_batch_k_means=True,
                      load_tfidf_vectorizer=True)


def search_tags(text):
    # category = model.predict(text)
    category = clasterer.predict(text)
    return list(map(lambda x:x['text'],searcher.search_tags(text, category)))


# start = time.time()
# print(search_tags('набор для шитья'))
# end = time.time()
# print(end - start)


