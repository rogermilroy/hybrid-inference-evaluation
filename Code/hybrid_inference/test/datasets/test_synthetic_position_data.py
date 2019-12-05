from unittest import TestCase
from src.data.synthetic_position_dataset import SyntheticPositionDataset
import torch


class TestSyntheticPositionData(TestCase):

    def setUp(self) -> None:
        self.x0 = torch.tensor([0., 1., 0., 2.])
        pass

    def tearDown(self) -> None:
        pass

    def test_n_samples(self) -> None:
        """
        Test that the dataset generates the correct number of samples. Also tests that the len function works correctly.
        :return: None
        """
        dataset = SyntheticPositionDataset(x0=self.x0, n_samples=100, sample_length=10, starting_point=0, seed=42)
        self.assertEqual(len(dataset), 100)

    def test_starting_point_len(self) -> None:
        """
        Test that the starting point doesn't interfere with the number of samples created
        :return: None
        """
        dataset = SyntheticPositionDataset(x0=self.x0, n_samples=100, sample_length=10, starting_point=100, seed=42)
        self.assertEqual(len(dataset), 100)

    def test_get_item(self) -> None:
        """
        Test the format of the items returned from the dataset.
        :return: None
        """
        dataset = SyntheticPositionDataset(x0=self.x0, n_samples=100, sample_length=10, starting_point=0, seed=42)
        sample, label = dataset[12]
        print(sample)
        print(label)

    def test_starting_point(self) -> None:
        """
        Test that the starting point actually creates different data.
        :return: None
        """
        dataset = SyntheticPositionDataset(x0=self.x0, n_samples=100, sample_length=10,  starting_point=0, seed=42)
        dataset1 = SyntheticPositionDataset(x0=self.x0, n_samples=100, sample_length=10, starting_point=100, seed=42)
        self.assertFalse(torch.allclose(dataset[1][0], dataset1[1][0], atol=1e-1))



