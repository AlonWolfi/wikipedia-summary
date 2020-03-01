import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Iterable

PROJECT_NAME = 'wikipedia-summary'
CONFIG_FILE = 'config.json'
CONFIG = None


def get_config():
    '''
    @return: The config json instance of the project
    '''
    global CONFIG
    # If config not created - creates config
    if not CONFIG:
        # get path
        from utils.utils import get_project_dir
        project_path = get_project_dir()
        config_path = project_path / CONFIG_FILE
        # loads config
        with open(config_path) as json_file:
            CONFIG = json.load(json_file)
    return CONFIG


def get_project_dir() -> Path:
    '''
    @return: The path of the project
    '''
    current_path = Path.cwd()
    while current_path.name != PROJECT_NAME:
        current_path = current_path.parent
    return current_path


def task_done(task_name):
    file_path = get_project_dir() / get_config()['cache_dir'] / (task_name + '.done')
    try:
        os.stat(file_path.parent)
    except:
        os.mkdir(file_path.parent)
    with open(file_path, 'w+') as file:
        file.write('')
    return True


def get_file_path_from_config(file_name: str, dir_name: str = 'cache_dir'):
    config = get_config()

    if dir_name == '':
        return get_project_dir() / config[file_name]

    return get_project_dir() / config[dir_name] / config[file_name]


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

    if file_path.suffix == '.pickle':
        if type(data) == pd.DataFrame:
            pd.to_pickle(data, file_path)
        else:
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


def read_data(file_path: Union[str, Path], is_dataframe = True, encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    validate_path(file_path)

    data = None

    if file_path.suffix == '.pickle':
        if is_dataframe:
            data = pd.read_pickle(file_path)
        else:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
    elif file_path.suffix == '.json':
        with open(file_path, 'rb') as file:
            data = json.load(file)
    else:
        with open(file_path, 'r+', encoding=encoding) as file:
            print(file_path)
            data = file.read()
    return data


if __name__ == '__main__':
    print('project dir is: ' + str(get_project_dir()))

    task_done('test')
    print('the task "test" is done!')
