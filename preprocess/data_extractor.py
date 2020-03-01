import os
import glob
import nltk
import gensim
import pandas as pd
import wikipedia
import wikipediaapi
import wptools

import utils.luigi_wrapper as luigi

from preprocess.page_list_extractor_wptools import PageListExtractorTask

from utils.wikipedia_utils import *
from utils.utils import *


class DataExtractor(luigi.Task):
    def requires(self):
        return PageListExtractorTask()

    def output(self):
        return luigi.LocalTarget(get_file_path('full_df.pickle'))

    def run(self):
        pages_lst = self.requires().get_output().split('\n')

        if self.DEBUG:
            pages_lst = pages_lst[:50]

        page_index = []
        content_lst = []
        infoboxes_lst = []
        for p in pages_lst:
            doc = load_text(p)
            if doc is None:
                continue
            infobox = load_info_box(p)
            if infobox is None:
                continue

            page_index.append(p)
            content_lst.append(doc)
            infoboxes_lst.append(infobox)

        df: pd.DataFrame() = pd.DataFrame()

        print(pages_lst)

        df['page'] = page_index
        df['text'] = content_lst
        df['infobox'] = infoboxes_lst
        df = df.set_index('page')

        print(df.index)

        save_data(df, self.output().path)


if __name__ == '__main__':
    luigi.run_task(DataExtractor)
