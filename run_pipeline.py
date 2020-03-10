import utils.luigi_wrapper as luigi

from utils.utils import get_from_config

from visualization.plotROC import PlotROCTask


class PipelineEnd(luigi.Task):
    def requires(self):
        pass


if __name__ == '__main__':
    luigi.run_task(PlotROCTask(), local_scheduler=get_from_config('luigi_local_scheduler'), delete_all=False)
