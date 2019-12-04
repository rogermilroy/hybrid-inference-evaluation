from unittest import TestCase
from ...src.training.train_hybrid_inference import train_hybrid_inference


class TestTraining(TestCase):

    def test_training(self):
        train_hybrid_inference(5, False, "./hybrid_inference_params.pt")
