import os

import pandas as pd
import json

from Back.Parser.config import DATA_DIR


def quantity_to_categories(x):
    if x < 200:
        return 0
    if x < 1000:
        return 1
    if x < 5000:
        return 2
    if x < 10000:
        return 3
    if x < 50000:
        return 4
    return 5


def convert_search_data(file_names_prefix="", nrows=1000000):
    data = pd.read_csv(os.path.join(DATA_DIR, 'search_history.csv'), nrows=nrows, ).drop(
        ['wbuser_id', 'locale', 'weekday', 'time'], axis=1)
    data = data.drop(data[data.cnt < 50].index)
    data = data.drop_duplicates(subset='UQ', keep="last")
    data.to_csv(os.path.join(DATA_DIR, f"{file_names_prefix}valid_data.csv"), encoding='utf-8')


def convert_query_popularity(file_names_prefix=""):
    data = pd.read_csv(os.path.join(DATA_DIR, 'query_popularity.csv'), encoding='utf-8', encoding_errors='ignore')
    # data = data.drop(data[data.query_popularity <= 5].index)
    data = data.drop_duplicates(subset='query', keep="last")
    data.to_csv(os.path.join(DATA_DIR, f"{file_names_prefix}valid_popularity_data.csv"), encoding='utf-8')


def replace_quantity_to_categories(file_names_prefix=""):
    data = pd.read_csv(os.path.join(DATA_DIR, f"{file_names_prefix}train_data.csv"), encoding='utf-8',
                       encoding_errors='ignore')
    data['cnt'] = data['cnt'].apply(quantity_to_categories)
    data.to_csv(os.path.join(DATA_DIR, f"{file_names_prefix}train_data.csv"), encoding='utf-8')


def convert_comparsion_data_to_json(file_names_prefix=""):
    data = pd.read_csv(os.path.join(DATA_DIR, f"{file_names_prefix}comparsion_data.csv"), encoding='utf-8',
                       encoding_errors='ignore')
    data = data.to_json(indent=4)
    data = json.loads(data)
    new_data = {}
    for key, value in data.items():
        for i, (el_key, el_value) in enumerate(value.items()):
            if not i in new_data:
                new_data[i] = {}
            new_data[i][key] = el_value
    valid_data = {}
    for value in new_data.values():
        if not value['Num_Cluster'] and str(value['Num_Cluster']) != '0':
            continue

        if not value['query']:
            continue

        value['Num_Cluster'] = int(value['Num_Cluster'])
        value['query_popularity'] = int(value['query_popularity'])
        if not value['Num_Cluster'] in valid_data:
            valid_data[value['Num_Cluster']] = {}

        if not value['query_popularity'] in valid_data[value['Num_Cluster']]:
            valid_data[value['Num_Cluster']][value['query_popularity']] = []

        if '/' in value['query']:
            valid_data[value['Num_Cluster']][value['query_popularity']].extend(list(map(lambda x: {
                'text': x.strip(),
            }, value['query'].split('/'))))
        else:
            strip_string_len = len(value['query'].strip().split(' '))
            if strip_string_len < 2:
                continue
            valid_data[value['Num_Cluster']][value['query_popularity']].append({
                'text': value['query'],
            })

    with open(os.path.join(DATA_DIR, f"{file_names_prefix}comparsion_data.json"), 'w', encoding='utf-8') as file:
        json.dump(valid_data, file, ensure_ascii=False, indent=4)


def convert_train_data_to_reserv_json(file_names_prefix=""):
    data = pd.read_csv(os.path.join(DATA_DIR, f"{file_names_prefix}train_data.csv"), encoding='utf-8',
                       encoding_errors='ignore')
    data = data.to_json(indent=4)
    data = json.loads(data)
    new_data = {}
    for key, value in data.items():
        for i, (el_key, el_value) in enumerate(value.items()):
            if not i in new_data:
                new_data[i] = {}
            new_data[i][key] = el_value
    valid_data = {}
    for value in new_data.values():
        if not value['Num_Cluster'] and str(value['Num_Cluster']) != '0':
            continue

        value['Num_Cluster'] = int(value['Num_Cluster'])

        if not value['Num_Cluster'] in valid_data:
            valid_data[value['Num_Cluster']] = []

        if '/' in value['UQ']:
            valid_data[value['Num_Cluster']].extend(list(map(lambda x: {
                'text': x.strip(),
                'quantity': value['cnt']
            }, value['UQ'].split('/'))))
        else:
            strip_string_len = len(value['UQ'].strip().split(' '))
            if strip_string_len < 2:
                continue
            valid_data[value['Num_Cluster']].append({
                'text': value['UQ'],
                'quantity': value['cnt']
            })

    for key in valid_data.keys():
        valid_data[key] = list(
            map(lambda x: {"text": x['text']},
                reversed(
                    sorted(
                        valid_data[key],
                        key=lambda x: x['quantity']
                    ))))[:min([100,len(valid_data[key])])]

    with open(os.path.join(DATA_DIR, f"{file_names_prefix}reserve_comparsion_data.json"), 'w',
              encoding='utf-8') as file:
        json.dump(valid_data, file, ensure_ascii=False, indent=4)

