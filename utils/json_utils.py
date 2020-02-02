import json
from pathlib import Path

from utils.utils import get_project_dir

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
        project_path = get_project_dir()
        config_path = project_path / CONFIG_FILE
        # loads config
        with open(config_path) as json_file:
            CONFIG = json.load(json_file)
    return CONFIG