from abc import ABC

class Predictor(ABC):

    def __init__(self):
        super().__init__()

    def predict(self, **kwargs):
        pass
