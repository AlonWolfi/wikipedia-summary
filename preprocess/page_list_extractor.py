import xml
import shutil
import xml.etree.ElementTree as ET
import re
import os
import utils.luigi_wrapper as luigi

from pathlib import Path

from utils.text_utils import *
from utils.xml_utils import *
from utils.utils import *

INSTANCE_LINK_RE = '^\*\[\[([^\]]*)\]\]'

# BUGFIX
# IS_DEPENDENCY = __name__ == '__main__'


class PageListValidation(luigi.Task):
    def output(self):
        return luigi.LocalTarget(get_file_path('page_list.xml', 'data'))

    def run(self):
        pass


class PageListExtractorTask(luigi.Task):
    '''
    This task gets
    '''

    def requires(self):
        return PageListValidation()

    def output(self):
        return luigi.LocalTarget(get_file_path('page_list.txt'))

    # def complete(self):
        # return IS_DEPENDENCY

    @staticmethod
    def _get_intance_name(line):
        '''
        @param line: line of instance
        @return: the link of the instance of the line
        '''
        try:
            possible_names = re.search(INSTANCE_LINK_RE, line).group(1)
            return possible_names.split('|')[0]
        except:
            # No link which means no link and no page in wikipedia
            return None

    def run(self):
        # Read xml
        data_list_xml_path = self.input().path

        remove_first_line_nonesense_xml(data_list_xml_path)

        data_list = list()

        # for each list in the lists of raw_text
        for _, page_text in generate_pages(data_list_xml_path):
            # Get line generator
            lines = page_text.split('\n')
            lines_generator = get_lines_generator(lines)
            line = next(lines_generator)

            # Get to main part of text (list)
            while not is_line_main_title(line):
                line = next(lines_generator)

            line = next(lines_generator)

            while not is_line_main_title(line):
                # Check if line is of instance
                if len(line) > 0 and line[0] == '*':
                    instance_name = self._get_intance_name(line)
                    if instance_name:
                        data_list.append(instance_name)
                line = next(lines_generator)

        save_data('\n'.join(data_list), self.output().path)


if __name__ == '__main__':
    luigi.run_task(PageListExtractorTask)
