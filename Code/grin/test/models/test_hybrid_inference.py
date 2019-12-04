from ...src.data.synthetic_position_dataset import SyntheticPositionDataset
from unittest import TestCase
import torch
from ...src.models.hybrid_inference_model import HybridInference


class TestHybridInference(TestCase):

    def setUp(self) -> None:
        self.F = torch.tensor([[1., 1., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 1.],
                          [0., 0., 0., 1.]])
        self.H = torch.tensor([[1., 0., 0., 0.],
                          [0., 0., 1., 0.]])
        self.Q = torch.tensor([[0.05 ** 2, 0., 0., 0.],
                          [0., 0.05 ** 2, 0., 0.],
                          [0., 0., 0.05 ** 2, 0.],
                          [0., 0., 0., 0.05 ** 2]])
        self.R = (0.05 ** 2) * torch.eye(2)
        self.model = HybridInference(F=self.F,
                                     H=self.H,
                                     Q=self.Q,
                                     R=self.R,
                                     gamma=1e-4)
        self.dataset = SyntheticPositionDataset(x0=torch.tensor([0., 0.1, 0., 0.1]), n_samples=100, sample_length=10,
                                                starting_point=0, seed=42)

    def tearDown(self) -> None:
        pass

    def test_forward(self):
        obs, states = self.dataset[4]
        xs = self.H.T.matmul(obs.T)
        result = self.model(obs, iterations=100)
        print(states - result.t())
