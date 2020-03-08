import utils.luigi_wrapper as luigi

from visualization.plotROC import PlotROCTask

class PipelineEnd(luigi.Task):
    def requires(self):
        pass


if __name__ == '__main__':
    x = PlotROCTask()
    luigi.run_task(x, local_scheduler=True, delete_all=False)
