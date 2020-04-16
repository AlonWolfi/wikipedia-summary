import numpy as np

import utils.luigi_wrapper as luigi

from extraction.wikipedia_list_extraction import WikipediaListExtractionTask

from utils.wikipedia_utils import *
from utils.utils import *


class DataExtractionTask(luigi.Task):
    output_path = 'full_df.pickle'

    def requires(self):
        return WikipediaListExtractionTask()

    def output(self):
        return luigi.LocalTarget(get_file_path(self.output_path, 'raw_data'))

    @staticmethod
    def _get_df(page_index, content_lst, infoboxes_lst):
        df: pd.DataFrame = pd.DataFrame()
        df['page'] = page_index
        df['text'] = content_lst
        df['infobox'] = infoboxes_lst
        df = df.set_index('page')
        return df

    def run(self):
        pages_lst = self.get_task_inputs().split('\n')

        if self.config['debug']['DEBUG']:
            pages_lst = pages_lst[:50]

        page_index = []
        content_lst = []
        infoboxes_lst = []

        bad_pages = []

        if self.config['extraction']['subcache']:
            cached_df = read_data(get_file_path(self.output_path, 'subcache'))
            if cached_df is not None:
                page_index = list(cached_df.index)
                content_lst = list(cached_df['text'])
                infoboxes_lst = list(cached_df['infobox'])

            bad_pages_file = read_data(get_file_path('bad_pages.txt', 'subcache'))
            if bad_pages_file is not None:
                bad_pages = bad_pages_file.split('\n')

        for p in pages_lst:
            if p in page_index or p in bad_pages:
                continue

            # cache old__data
            if self.config['extraction']['subcache']:
                if np.random.randint(50) == 0:
                    df_cache = self._get_df(page_index, content_lst, infoboxes_lst)
                    save_data(df_cache, get_file_path(self.output_path, 'subcache'))
                    save_data('\n'.join(bad_pages), get_file_path('bad_pages.txt', 'subcache'))

            try:
                doc = load_text(p)
            except:
                bad_pages.append(p)
                continue

            if doc is None:
                bad_pages.append(p)
                continue

            try:
                infobox = load_info_box(p)
            except:
                bad_pages.append(p)
                continue

            if infobox is None:
                bad_pages.append(p)
                continue

            page_index.append(p)
            content_lst.append(doc)
            infoboxes_lst.append(infobox)

        df = self._get_df(page_index, content_lst, infoboxes_lst)
        save_data(df, self.output().path)


if __name__ == '__main__':
    luigi.run_task(DataExtractionTask())
