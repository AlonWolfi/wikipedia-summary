import re
import os
import json
from pathlib import Path
from typing import Union, Iterable

from utils.utils import get_project_dir


def get_lines_generator(lines):
    '''
    yields each line of text
    '''
    for line in lines:
        yield line


def is_line_main_title(line):
    '''
    @param line: line in wikipedia
    @return: True if line is a main title
    '''
    return (line[:2] == '==' and line[-2:] == '==') and (line[:3] != '===' and line[-3:] != '===')


def save_data(data, file_path: Union[str, Path], encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    for dir in file_path.parents:
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)

    with open(file_path, 'w+', encoding=encoding) as file:
        if file_path.suffix == '.json':
            json.dump(data, file)
        else:
            file.write(data)
