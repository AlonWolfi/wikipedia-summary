import wptools
import utils.luigi_wrapper as luigi
from utils.utils import *


class WikipediaListExtractorTask(luigi.Task):
    '''
    This task crawls through a Category within wikipedia and will make a list of all wikipedia pages.
    If there are sub-categories in the categories it will crawl inside them and will return all pages.
    @param initial_category: The category within to do the crawl
    @return: saves a txt file with the list of all pages
    '''

    # The category within to do the crawl
    initial_category = luigi.Parameter(default='Category:Musicians')

    def output(self):
        return luigi.LocalTarget(get_file_path('wikipedia_page_list.txt', 'raw_data'))

    def __get_category(self, category_name):
        '''
        A recursion function that gets a category and returns all it's pages within it.
        If there are subcategories will recursively run the function
        Will add all pages to self.pages (set)
        @return: None
        '''
        # Gets the old__data within the wiki page
        wiki_data = wptools.category(category_name).get_members().data

        try:
            # tries to get subcatgories
            subcategories = wiki_data['subcategories']
        except:
            # Stop Condition - If there are no subcategories
            # Loops through all members (pages) in category
            # adds them to pages
            for member in wiki_data['members']:
                print(f"Added to Wikipedia pages list : {member['title']}")
                self.pages.add(member['title'])

                # cache results
                if self.config['extraction']['subcache']:
                    if len(self.pages) % 50 == 0:
                        save_data('\n'.join(self.pages), self.output().path)
            return

        # For DEBUG - smaller subcategories
        if self.config['debug']['DEBUG']:
            subcategories = subcategories[:20]

        for cat in subcategories:
            print(f"Entered to Wikipedia Category : {cat['title']}")
            self.__get_category(cat['title'])

    def run(self):
        self.pages = set()

        self.__get_category(self.initial_category)

        save_data('\n'.join(self.pages), self.output().path)


if __name__ == '__main__':
    luigi.run_task(WikipediaListExtractorTask())
