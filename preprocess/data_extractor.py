import xml
import shutil
import xml.etree.ElementTree as ET
import re
import luigi
import json

from luigi import Task, LocalTarget
from pathlib import Path

from utils.json_utils import get_config
from utils.text_utils import *
from utils.xml_utils import *
from utils.utils import get_project_dir, task_done

from preprocess.data_list_extractor import DataListExctratorTask


class InstanceDataExtractorTask(Task):
    name = luigi.Parameter()
    page_text = luigi.Parameter()

    def requires(self):
        pass

    def output(self):
        config = get_config()

        data_path = get_project_dir() / get_config()['data_dir']
        raw_data_path = data_path / get_config()['data_raw_dir']
        labels_path = data_path / get_config()['data_labels_dir']
        return {
            'raw_text': LocalTarget(raw_data_path / (self.name + '.txt')),
            'labels': LocalTarget(labels_path / (self.name + '.json'))
        }

    @staticmethod
    def _get_lines_generator(lines):
        for line in lines:
            yield line

    @staticmethod
    def extract_answer(answer: str):
        start_link = '\[\['
        link = '[^\[\|]+'
        links = '((' + link + '\|)*' + '(' + link + '))'
        end_link = '\]\]'

        def get_link_first(m):
            try:
                return m.group(2)[:-1]
            except:
                return m.group(1)

        return re.sub(start_link + links + end_link, get_link_first, answer)

    @classmethod
    def extract_label(cls, label_line):
        # [1:] for no '|'
        line_splited = label_line[1:].split('=')
        question = line_splited[0]
        answer = '='.join(line_splited[1:])
        if answer == '':
            return None
        return question, cls.extract_answer(answer)

    def save(self, data, labels):
        save_data(data, self.output()['raw_text'].path)
        save_data(labels, self.output()['labels'].path)

    def run(self):
        lines = self.page_text.split('\n')
        lines_generator = get_lines_generator(lines)

        labels = list()

        line = next(lines_generator)


        for line in lines_generator:
            if line[:9] == '{{Infobox':
                break
        try:
            line = next(lines_generator)
        except:
            labels = dict()
            data = ''
            self.save(data, labels)
            return False


        labels = dict()

        for line in lines_generator:
            if re.match('^}}.*',line):
                break

            extraction = self.extract_label(line)
            if extraction:
                q, a = extraction
                labels[q] = a

        data = '\n'.join(list(lines_generator))

        self.save(data, labels)


class DataExctratorTask(Task):

    def _file_to_list(self, file_path):
        instance_list = list()
        with open(file_path) as file:
            instance_list = file.readlines().split('/n')
        return instance_list

    def requires(self):
        return {
            'data_list': DataListExctratorTask()
        }

    def output(self):
        return LocalTarget(Path(get_config()['cache_dir']) / (self.__class__.__name__ + '.done'))

    def run(self):
        data_path = get_project_dir() / get_config()['data_dir'] / 'data.xml'
        remove_first_line_nonesense_xml(data_path)

        for name, page_text in generate_pages(data_path):
            # Call Task
            yield InstanceDataExtractorTask(name=name, page_text=page_text)

        task_done(str(self.__class__.__name__))


if __name__ == '__main__':
    import os

    task_done_path = get_project_dir() / get_config()['cache_dir'] / 'DataExctratorTask.done'
    if os.path.exists(task_done_path):
        os.remove(task_done_path)
    luigi.build(
        [
            DataExctratorTask()
        ],
        local_scheduler=False
    )
