from unittest import TestCase
from src.models.linear_model import ConstantVelocityModel
import torch


class TestLinearModel(TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x0 = torch.tensor([0., 1., 0., 2.])
        self.model = ConstantVelocityModel(x0=self.x0, lambdasq=0.0001, sigma_x=0.0001, sigma_y=0.0001)

    def tearDown(self) -> None:
        pass

    def test_one_cycle(self):
        state, meas = self.model()
        print(state)
        print(meas)
        ref_state = torch.tensor([1.0034, 1.0013, 2.0023, 2.0023])
        ref_meas = torch.tensor([0.9921, 2.0005])
        self.assertTrue(torch.allclose(state, ref_state, atol=1e-4))
        self.assertTrue(torch.allclose(meas, ref_meas, atol=1e-4))


