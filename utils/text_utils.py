import re
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


def save_to_file(txt: Union[Iterable[str], str], file_path: Union[str, Path], encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(txt) == str:
        txt = [txt]

    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    with open(file_path, 'w', encoding=encoding) as file:
        for line in txt:
            file.write(line + '\n')
