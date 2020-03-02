import wptools
import utils.luigi_wrapper as luigi
from utils.utils import *


class PageListExtractorTask(luigi.Task):
    '''
    This task gets
    '''

    def output(self):
        return luigi.LocalTarget(get_file_path('page_list_wptools.txt'))

    def __get_category(self, category_name):
        data = wptools.category(category_name).get_members().data
        try:
            subcategories = data['subcategories']
        except:
            for member in data['members']:
                print(member['title'])
                self.pages.add(member['title'])

                # cache results
                if get_from_config('SUBCACHE'):
                    if len(self.pages) % 50 == 0:
                        save_data('\n'.join(self.pages), self.output().path)
            return

        if self.DEBUG:
            subcategories = subcategories[:20]

        for cat in subcategories:
            print(cat['title'])
            self.__get_category(cat['title'])

    def run(self):
        self.pages = set()

        starting_category = 'Category:Musicians by band'
        self.__get_category(starting_category)

        save_data('\n'.join(self.pages), self.output().path)


if __name__ == '__main__':
    luigi.run_task(PageListExtractorTask)
