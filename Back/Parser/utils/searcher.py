import json
import os
import random
import traceback
from pprint import pprint

from Back.Parser.config import DATA_DIR
from Back.Parser.utils.limiter import limiter


class Searcher:

    def __init__(self,file_names_prefix="",load_comparsion_data=True,load_reserve_comparsion_data=True):
        self.file_names_prefix = file_names_prefix

        if load_comparsion_data:
            try:
                self.load_comparsion_data()
                print('Данные загружены')
            except:
                print('Невозможно загрузить данные')
                traceback.print_exc()
                self.comparsion_data = None

        if load_reserve_comparsion_data:
            try:
                self.load_reserve_comparsion_data()
                print('Данные загружены')
            except:
                print('Невозможно загрузить данные')
                traceback.print_exc()
                self.reserve_comparsion_data = None

    def load_comparsion_data(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}comparsion_data.json"),'r',encoding='utf-8') as file:
            self.comparsion_data = json.load(file)

    def load_reserve_comparsion_data(self):
        with open(os.path.join(DATA_DIR, f"{self.file_names_prefix}reserve_comparsion_data.json"),'r',encoding='utf-8') as file:
            self.reserve_comparsion_data = json.load(file)

    def search_similar(self):
        pass

    def search_tags(self, text, category):
        limit_text = limiter([text])[0]
        usage_tags = []
        need_tag_counter=10

        current_compare_texts = self.comparsion_data.get(str(category))
        # if not category
        if current_compare_texts:
            tags_sum = 0
            for row in current_compare_texts.values():
                # print(row)
                tags_sum += len(row)
                if tags_sum > 10:
                    break

            for i in range(1,11)[::-1]:
                if not current_compare_texts.get(str(i)):
                    continue
                while (len(current_compare_texts[str(i)]) and need_tag_counter):
                    usage_tags.append(current_compare_texts[str(i)].pop(random.randint(0,len(current_compare_texts[str(i)])-1)))
                    need_tag_counter-=1
                if not need_tag_counter:
                    break
            if not need_tag_counter:
                return usage_tags
        current_compare_texts = self.reserve_comparsion_data.get(str(category))
        while (len(current_compare_texts) and need_tag_counter):
            usage_tags.append(
                current_compare_texts.pop(0))
            need_tag_counter -= 1

        return usage_tags





