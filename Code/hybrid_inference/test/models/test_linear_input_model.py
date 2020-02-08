from unittest import TestCase
from src.models.linear_input_model import ConstantVelocityWInputModel
import torch


class TestLinearInputModel(TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x0 = torch.tensor([0., 1., 0., 2.])
        self.model = ConstantVelocityWInputModel(x0=self.x0, lambdasq=0.0001, sigma_x=0.0001,
                                                 sigma_y=0.0001)

    def tearDown(self) -> None:
        pass

    def test_one_cycle(self):
        state, meas = self.model()
        print(state)
        print(meas)
        ref_state = torch.tensor([1.0, 1.0, 2.5, 3.0])
        ref_meas = torch.tensor([1.0, 2.5])
        self.assertTrue(torch.allclose(state, ref_state, atol=1e-2))
        self.assertTrue(torch.allclose(meas, ref_meas, atol=1e-2))
