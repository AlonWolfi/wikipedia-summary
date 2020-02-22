import json
from pathlib import Path
from typing import Union, Iterable



CONFIG_FILE = 'config.json'
CONFIG = None

def save_dict_to_json(data: dict, file_path: Union[str, Path]):
    try:
        if file_path[-5:] != '.json':
            raise Exception
    except:
        print('JSON file must end with .json')

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

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