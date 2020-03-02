import utils.luigi_wrapper as luigi

from models.questions_model import QuestionsModel

if __name__ == '__main__':
    luigi.run_task(QuestionsModel, local_scheduler=True, delete_all=False)
