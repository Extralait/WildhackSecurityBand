import os

import pandas as pd
from pymystem3 import mystem

from Back.Parser.config import DATA_DIR


def limiter(texts):
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    txtpart = lol(texts, 10000)
    res = []
    for txtp in txtpart:
        alltexts = ' '.join([txt.lower() + ' br ' for txt in txtp])

        words = mystem.Mystem().lemmatize(text=alltexts)
        doc = []
        for txt in words:
            if txt != '\n' and txt.strip() != '':
                if txt == 'br':
                    res.append(doc)
                    doc = []
                else:
                    doc.append(txt)
    return res


def limit_comparsion_data(file_names_prefix):
    data = pd.read_csv(os.path.join(DATA_DIR, f"{file_names_prefix}clastered_popularity_data.csv"))

    text = limiter(data['query'].apply(lambda x:str(x).lower()).to_list())

    df2 = {"limited_query": list(map(lambda x: ' '.join(x), text))}
    dfMBK = pd.DataFrame(df2)
    df = pd.concat([data, dfMBK], axis=1)

    df.to_csv(os.path.join(DATA_DIR, f"{file_names_prefix}comparsion_data.csv"), index=False)
