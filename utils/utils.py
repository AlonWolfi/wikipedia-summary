import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Union

from matplotlib.pyplot import Figure

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


def get_from_config(element: str, category: str = None):
    if category:
        return get_config()[category][element]
    else:
        return get_config()[element]


def get_file_path(file_name: str, dir_names: Union[str, list] = []):
    if type(dir_names) == str:
        dir_names = [dir_names]
    CACHE_FOLDER = get_project_dir() / 'cache'
    current_dir = CACHE_FOLDER
    for dir in dir_names:
        current_dir /= str(dir)
    return current_dir / file_name


def validate_path(file_path: Union[str, Path]):
    for folder in list(file_path.parents)[::-1]:
        try:
            os.stat(folder)
        except:
            os.mkdir(folder)


def loop_through_iterable(iterable, func_for_ins):
    if type(iterable) == dict:
        outputs = dict()
        for k, v in iterable.items():
            outputs[k] = loop_through_iterable(v, func_for_ins)
        return outputs
    elif type(iterable) == list:
        outputs = list()
        for ele in iterable:
            outputs.append(loop_through_iterable(ele, func_for_ins))
        return outputs
    elif type(iterable) == set:
        outputs = set()
        for ele in iterable:
            outputs.add(loop_through_iterable(ele, func_for_ins))
        return outputs
    else:
        return func_for_ins(iterable)


def loop_through_iterables_list(iterables_list, func_for_ins):
    # TODO - finish it sometimes - for now leave alone
    # Check all are the same type
    iterable_type = type(iterables_list[0])
    if any([type(iterable) != iterable_type for iterable in iterables_list]):
        print(f'Iterables list must be in the same type : {iterable_type}')
        raise Exception
    if iterable_type == dict:
        outputs_list = [dict() for i in iterables_list]
        # for kvs in zip(*[d.items() for d in iterables_list]):
        #     for i, (k, v) in enumerate(kvs):
        #
        #     outputs_list[k] = loop_through_iterables_list(v, func_for_ins)
        return outputs_list
    elif iterable_type == list:
        outputs = list()
        # for ele in iterable:
        #     outputs.append(loop_through_iterable(ele, func_for_ins))
        return outputs
    elif iterable_type == set:
        outputs = set()
        # for ele in iterable:
        #     outputs.add(loop_through_iterable(ele, func_for_ins))
        return outputs
    # else:
    #     return func_for_ins(iterable)


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True


def save_data(data, file_path: Union[str, Path], encoding: str = "utf-8"):
    '''
    Saves text to file
    '''
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    validate_path(file_path)

    if os.path.exists(file_path):
        os.remove(file_path)
        # to fix writing after deleting
        # TODO - Fix deletion without reading
        time.sleep(10)

    elif file_path.suffix == '.pickle':
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    elif file_path.suffix == '.json':
        with open(file_path, 'wb') as file:
            json.dump(data, file)
    elif type(data) == Figure:
        data.save_fig(file_path)
    else:
        with open(file_path, 'w+', encoding=encoding) as file:
            file.write(data)


def read_data(file_path: Union[str, Path], encoding: str = "utf-8"):
    """
    Saves data to file_path
    @param file_path:
    @param encoding:
    @return:
    """
    if type(file_path) == str:
        file_path = get_project_dir() / file_path

    validate_path(file_path)

    data = None

    if not os.path.exists(file_path):
        warnings.warn(f'Warning:   File not found: {file_path}', Warning)
        return None

    if file_path.suffix == '.pickle':
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    elif file_path.suffix == '.json':
        with open(file_path, 'rb') as file:
            data = json.load(file)
    elif file_path.suffix == '.jpg' or file_path.suffix == '.png':
        data = plt.imread(file_path, format='PNG')
    else:
        with open(file_path, 'r+', encoding=encoding) as file:
            data = file.read()
    return data
