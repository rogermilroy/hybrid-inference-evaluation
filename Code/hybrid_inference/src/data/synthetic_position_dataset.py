from torch.utils import data
from torch import tensor
import torch
from src.models.linear_model import ConstantVelocityModel


class SyntheticPositionDataset(data.Dataset):
    """
    A class that creates and returns synthetic position data from the GRIN to train and evaluate on.
    Data is created by transition functions.
    An unusual dataset as data is generated on demand rather than read from file.
    """

    def __init__(self, x0: tensor, n_samples: int = 5000, sample_length: int = 100, starting_point:int = 0, seed: int = 42, device='cpu'):
        """
        Initialises various parameters of the dataset.
        Each sample consists of a sequence of measurements and ground truths of length sample length
        :param n_samples: How many samples the dataset will contain. Default 5000
        :param seed: A random seed. This must be different for train and test sets. Default 42
        :param device: Which device the samples should be put on, can be cuda if available. Default cpu.
        """
        rawlabels, rawdata = self.generate_data(x0, n_samples, sample_length, starting_point, seed)
        dat = list()
        lab = list()
        for i in range(n_samples):
            dat.append(rawdata[i*sample_length: (i+1)*sample_length])
            lab.append(rawlabels[i*sample_length:(i+1)*sample_length])
        self.data = torch.stack(dat)
        self.labels = torch.stack(lab)

    def __len__(self):
        """
        Returns the length of the dataset
        :return: int: The length of the dataset.
        """
        return self.data.size()[0]

    def __getitem__(self, index):
        """
        Returns the data item at index index. The format of the item will be,
        :param index:
        :return:
        """
        return self.data[index], self.labels[index]

    @staticmethod
    def generate_data(x0: tensor, n_samples: int, sample_length: int, starting_point: int, seed: int) -> tensor:
        """
        Generates a dataset of n samples
        :param x0:
        :param n_samples:
        :param starting_point:
        :param seed:
        :return:
        """
        # create a linear model with mostly default parameters.
        torch.manual_seed(seed)
        model = ConstantVelocityModel(x0=x0)

        total_sample_size = n_samples*sample_length + starting_point

        ground_truth = torch.zeros((total_sample_size, x0.size()[0]))  # TODO check x0 dimensions.
        measurements = torch.zeros((total_sample_size, 2))  # TODO auto fill size of the measurements

        for i in range(total_sample_size):
            x, z = model()
            ground_truth[i, :] = x
            measurements[i, :] = z

        return ground_truth[starting_point:], measurements[starting_point:]
