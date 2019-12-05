import numpy as np
import torch
from src.data_utils import converters
import unittest


class TestConverters(unittest.TestCase):

    def setUp(self) -> None:
        # define a range of different arrays and tensors to test.
        oned = np.arange(128., dtype=np.float64)
        self.nump = [oned,
                     oned.reshape((2, -1)),
                     oned.reshape((2, 2, -1)),
                     oned.reshape((2, 2, 2, -1))]

        d = torch.arange(128., dtype=torch.float64)
        self.tor = [d,
                    d.reshape((2, -1)),
                    d.reshape((2, 2, -1)),
                    d.reshape((2, 2, 2, -1))]

    def tearDown(self) -> None:
        pass

    def test_numpy_to_torch(self):
        for n, t in zip(self.nump, self.tor):
            self.assertTrue(torch.allclose(converters.np2torch(n), t))

    def test_different_dtypes(self):
        for n, t in zip(self.nump, self.tor):
            n = n.astype(np.float32)
            t = t.to(torch.float32)
            self.assertTrue(torch.allclose(converters.np2torch(n), t))
            n = n.astype(np.int32)
            t = t.to(torch.int32)
            self.assertTrue(torch.allclose(converters.np2torch(n), t))

    def test_torch_to_numpy(self):
        for n, t in zip(self.nump, self.tor):
            self.assertTrue(np.allclose(converters.torch2np(t), n))

    def test_dtypes_torch_numpy(self):
        for n, t in zip(self.nump, self.tor):
            n = n.astype(np.float32)
            t = t.to(torch.float32)
            self.assertTrue(np.allclose(converters.torch2np(t), n))
            n = n.astype(np.int32)
            t = t.to(torch.int32)
            self.assertTrue(np.allclose(converters.torch2np(t), n))

    def test_both_ways(self):
        for n, t in zip(self.nump, self.tor):
            self.assertTrue(np.allclose(converters.torch2np(converters.np2torch(n)), n))
            self.assertTrue(torch.allclose(converters.np2torch(converters.torch2np(t)), t))


