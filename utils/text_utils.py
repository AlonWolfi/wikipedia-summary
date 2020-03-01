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