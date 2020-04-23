from unittest import TestCase

from src.data.gazebo_dataset import GazeboDataset


class TestGazeboPositionData(TestCase):
    """
    A minimal test class for the dataset.
    """

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_n_samples(self) -> None:
        """
        Test that the len function works correctly.
        :return: None
        """
        dataset = GazeboDataset(8, "../../../catkin_ws/recording/", 2)
        self.assertEqual(len(dataset), 6)

    def test_get_item(self) -> None:
        """
        Test the format of the items returned from the dataset.
        :return: None
        """
        dataset = GazeboDataset(8, "../../../catkin_ws/recording/", 2)
        ys, Fs, gt = dataset[1]
        print(ys)
        print(Fs)
        print(gt)
