import xml
import shutil
import xml.etree.ElementTree as ET
import re
import luigi

from luigi import Task, LocalTarget
from pathlib import Path

from utils.json_utils import get_config
from utils.text_utils import *
from utils.xml_utils import *
from utils.utils import get_project_dir

INSTANCE_LINK_RE = '^\*\[\[([^\]]*)\]\]'


class DataListExctratorTask(Task):
    '''
    This task gets
    '''

    def requires(self):
        pass

    def output(self):
        data_path = Path(get_config()['data_dir'])
        data_list_path = data_path / get_config()['data_list_file']
        return LocalTarget(data_list_path)

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

    def _get_xml_data_list_path(self):
        data_path = get_config()['data_dir']
        return get_project_dir() / data_path / get_config()['data_list_xml_file']

    def run(self):
        # Read xml
        data_list_xml_path = self._get_xml_data_list_path()

        remove_first_line_nonesense_xml(data_list_xml_path)

        data_list = list()

        # for each list in the lists of data
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

        save_to_file(data_list, self.output().path)


if __name__ == '__main__':
    luigi.build(
        [
            DataListExctratorTask()
        ],
        local_scheduler=True
    )
