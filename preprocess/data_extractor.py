import os
import glob
import nltk
import gensim
import pandas as pd
import wikipedia
import wikipediaapi
import wptools

import utils.luigi_wrapper as luigi

from preprocess.page_list_extractor import PageListExtractorTask

from utils.wikipedia_utils import *
from utils.utils import *


class DataExtractor(luigi.Task):
    def requires(self):
        return PageListExtractorTask()

    def output(self):
        return luigi.LocalTarget(get_file_path_from_config('data_file'))

    def run(self):
        pages_lst = self.requires().get_output()

        if self.DEBUG:
            pages_lst = pages_lst[:3]

        content_lst = []
        infoboxes_lst = []
        for p in pages_lst:
            doc = load_text(p)
            if doc is None:
                continue
            infobox = load_info_box(p)

            content_lst.append(doc)
            infoboxes_lst.append(infobox)

        df = pd.DataFrame()
        df['page'] = pages_lst
        df['text'] = content_lst
        df['infobox'] = infoboxes_lst
        df = df.set_index('page')

        save_data(df, self.output().path)


if __name__ == '__main__':
    luigi.build(
        [
            DataExtractor(
                DEBUG=True
            )
        ],
        local_scheduler=True
    )
    print('#### output ####')
    print(DataExtractor.get_output())
