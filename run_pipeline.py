import luigi

from preprocess.page_list_extractor import DataListExctratorTask
from preprocess.data_extractor import get_full_wikipedia_data
from preprocess.page_list_extractor import DataListExctratorTask

def run_data_list_extractor():
    luigi.build(
        [
            DataListExctratorTask()
        ],
        local_scheduler=False
    )

if __name__ == '__main__':
    run_data_list_extractor()
    full_df = get_full_wikipedia_data()


