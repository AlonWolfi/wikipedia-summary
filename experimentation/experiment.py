class Experiment:

    def __init__(self, params, y_pred, y_pred_after_prior, train_test_split, scores):
        self.params: dict = params
        self.y_pred = y_pred
        self.y_pred_after_prior = y_pred_after_prior
        self.train_test_split = train_test_split
        self.scores = scores

    def to_json(self):
        return {
            'params': self.params,
            'scores': self.scores
        }