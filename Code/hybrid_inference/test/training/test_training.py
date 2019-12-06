from unittest import TestCase
import torch
from src.training.train_hybrid_inference import train_hybrid_inference
from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataset import SyntheticPositionDataset


class TestTraining(TestCase):

    def setUp(self) -> None:
        self.H = torch.tensor([[1., 0., 0., 0.],
                               [0., 0., 1., 0.]])
        self.dataset = SyntheticPositionDataset(x0=torch.tensor([0., 0.1, 0., 0.1]), n_samples=100, sample_length=10, starting_point=0, seed=42)

    def test_training(self):
        train_hybrid_inference(5, False, "./hybrid_inference_params.pt")

    def test_trained_model(self):
        model = HybridInference(F=torch.tensor([[1., 1., 0., 0.],
                                                [0., 1., 0., 0.],
                                                [0., 0., 1., 1.],
                                                [0., 0., 0., 1.]]),
                                H=torch.tensor([[1., 0., 0., 0.],
                                                [0., 0., 1., 0.]]),
                                Q=torch.tensor([[0.05 ** 2, 0., 0., 0.],
                                                [0., 0.05 ** 2, 0., 0.],
                                                [0., 0., 0.05 ** 2, 0.],
                                                [0., 0., 0., 0.05 ** 2]]),
                                R=(0.05 ** 2) * torch.eye(2),
                                gamma=1e-4)
        model.load_state_dict(torch.load("./hybrid_inference_params.pt"))
        obs, states = self.dataset[4]
        xs = self.H.T.matmul(obs.T)
        best_estimate = model(obs, 100)
        print(states - best_estimate.t())

    def test_continue_training(self):
        train_hybrid_inference(1, False, "./hybrid_inference_params.pt", "./hybrid_inference_params.pt")

