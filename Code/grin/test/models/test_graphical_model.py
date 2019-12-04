from ...src.data.synthetic_position_dataset import SyntheticPositionDataset
from unittest import TestCase
import torch
from ...src.models.graphical_model import KalmanGraphicalModel


class TestGraphicalModel(TestCase):

    def setUp(self) -> None:
        self.H = torch.tensor([[1., 0., 0., 0.],
                               [0., 0., 1., 0.]])
        self.dataset = SyntheticPositionDataset(x0=torch.tensor([0., 0.1, 0., 0.1]), n_samples=100, sample_length=10, starting_point=0, seed=42 )
        self.model = KalmanGraphicalModel(F=torch.tensor([[1., 1., 0., 0.],
                                                          [0., 1., 0., 0.],
                                                          [0., 0., 1., 1.],
                                                          [0., 0., 0., 1.]]),
                                          H=torch.tensor([[1., 0., 0., 0.],
                                                          [0., 0., 1., 0.]]),
                                          Q=torch.tensor([[0.05**2, 0., 0., 0.],
                                                          [0., 0.05**2, 0., 0.],
                                                          [0., 0., 0.05**2, 0.],
                                                          [0., 0., 0., 0.05**2]]),
                                          R=(0.05**2) * torch.eye(2),
                                          standalone=True)

    def tearDown(self) -> None:
        pass

    def test_once(self):
        obs, states = self.dataset[3]
        xs = self.H.T.matmul(obs.T)
        result = self.model.once(xs, obs)
        print(result)

    def test_forward(self):
        obs, states = self.dataset[4]
        xs = self.H.T.matmul(obs.T)
        best_estimate = self.model(xs, obs, 1e-4, 100)
        print(states - best_estimate.t())
        self.assertTrue(torch.allclose(states, best_estimate.t(), atol=0.9))

