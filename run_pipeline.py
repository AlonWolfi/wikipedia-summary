import utils.luigi_wrapper as luigi

from visualization.plotROC import plotROCTask

class PipelineEnd(luigi.Task):
    def requires(self):
        pass


if __name__ == '__main__':
    luigi.run_task(plotROCTask(), local_scheduler=True, delete_all=False)
