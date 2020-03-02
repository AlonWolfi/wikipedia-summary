import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Iterable

PROJECT_NAME = 'wikipedia-summary'
CONFIG_FILE = 'config.json'
CONFIG = None

import matplotlib.pyplot as plt



def get_project_dir() -> Path:
    '''
    @return: The path of the project
    '''
    current_path = Path.cwd()
    while current_path.name != PROJECT_NAME:
        current_path = current_path.parent
    return current_path


def get_config():
    '''
    @return: The config json instance of the project
    '''
    global CONFIG
    # If config not created - creates config
    if not CONFIG:
        # get path
        project_path = get_project_dir()
        config_path = project_path / CONFIG_FILE
        # loads config
        with open(config_path) as json_file:
            CONFIG = json.load(json_file)
    return CONFIG


def get_from_config(element: str):
    return get_config()[element]


def get_file_path(file_name: str, dir_name: str = 'cache'):
    if dir_name == '':
        return get_project_dir() / file_name

    return get_project_dir() / dir_name / file_name


def validate_path(file_path: Union[str, Path]):
    for dir in list(file_path.parents)[::-1]:
        try:
            os.stat(dir)
        except:
            os.mkdir(dir)


def save_data(data, file_path: Union[str, Path], encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    validate_path(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    if file_path.suffix == '.pickle':
        with open(file_path, 'wb') as file:
            print(file_path)
            print(file)
            pickle.dump(data, file)
    elif file_path.suffix == '.json':
        with open(file_path, 'wb') as file:
            json.dump(data, file)
    else:
        with open(file_path, 'w+', encoding=encoding) as file:
            file.write(data)


def read_data(file_path: Union[str, Path], is_dataframe=True, encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    validate_path(file_path)

    data = None

    if file_path.suffix == '.pickle':
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    elif file_path.suffix == '.json':
        with open(file_path, 'rb') as file:
            data = json.load(file)
    elif file_path.suffix == '.jpg' or file_path.suffix == '.png':
        data = plt.imread(file_path)
    else:
        with open(file_path, 'r+', encoding=encoding) as file:
            print(file_path)
            data = file.read()
    return data
