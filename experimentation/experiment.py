class Experiment:

    def __init__(self, params, results):
        self._params: dict = params
        self._results: dict = results

    def to_json(self):
        return {
            'params': self.params,
            'results': self.results
        }