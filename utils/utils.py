import os

from pathlib import Path


PROJECT_NAME = 'wikipedia-summary'


def get_project_dir() -> Path:
    '''
    @return: The path of the project
    '''
    current_path = Path.cwd()
    while current_path.name != PROJECT_NAME:
        current_path = current_path.parent
    return current_path


def task_done(task_name):
    from utils.json_utils import get_config
    file_path = get_project_dir() / get_config()['cache_dir'] / (task_name + '.done')
    try:
        os.stat(file_path.parent)
    except:
        os.mkdir(file_path.parent)
    with open(file_path, 'w+') as file:
        file.write('')
    return True


if __name__ == '__main__':
    print('project dir is: ' + str(get_project_dir()))

    task_done('test')
    print('the task "test" is done!')