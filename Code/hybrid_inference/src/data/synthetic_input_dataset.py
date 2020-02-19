from torch.utils import data
from torch import tensor
import torch
from src.models.linear_input_model import ConstantVelocityWInputModel


class SyntheticInputDataset(data.Dataset):
    """
    A class that creates and returns synthetic position data from the GRIN to train and evaluate on.
    Data is created by transition functions.
    An unusual dataset as data is generated on demand rather than read from file.
    """

    def __init__(self, x0: tensor, n_samples: int = 5000, sample_length: int = 100,
                 starting_point: int = 0, seed: int = 42, device='cpu'):
        """
        Initialises various parameters of the dataset.
        Each sample consists of a sequence of measurements and ground truths of length sample length
        :param n_samples: How many samples the dataset will contain. Default 5000
        :param seed: A random seed. This must be different for train and test sets. Default 42
        :param device: Which device the samples should be put on, can be cuda if available. Default cpu.
        """
        rawlabels, rawdata, rawu = self.generate_data(x0, n_samples, sample_length, starting_point,
                                                  seed)
        dat = list()
        lab = list()
        u = list()
        for i in range(n_samples):
            dat.append(rawdata[i*sample_length: (i+1)*sample_length])
            lab.append(rawlabels[i*sample_length:(i+1)*sample_length])
            u.append(rawu[i*sample_length:(i+1)*sample_length])
        self.data = torch.stack(dat)
        self.labels = torch.stack(lab)
        self.u = torch.stack(u)

    def __len__(self):
        """
        Returns the length of the dataset
        :return: int: The length of the dataset.
        """
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, index):
        """
        Returns the data item at index index. The format of the item will be,
        :param index:
        :return:
        """
        return self.data[index], self.labels[index], self.u[index]

    @staticmethod
    def generate_data(x0: tensor, n_samples: int, sample_length: int, starting_point: int,
                      seed: int, input_fn=None) -> tensor:
        """
        Generates a dataset of n samples
        :param x0: Initial state vector.
        :param n_samples: The number of samples desired.
        :param sample_length: The length of the samples.
        :param starting_point: The number of cycles from the initial state after which samples
        will be taken.
        :param seed: A seed for the random number generator, for reproducibility.
        :param input_fn: A generator function for the inputs, takes parameter t returns a 2
        vector of x and y accelerations. (t) => [ax, ay]
        :return:
        """
        # create a linear model with mostly default parameters.
        torch.manual_seed(seed)
        model = ConstantVelocityWInputModel(x0=x0, input_fn=input_fn)

        total_sample_size = n_samples*sample_length + starting_point*sample_length

        ground_truth = torch.zeros((total_sample_size, x0.size()[0]))
        measurements = torch.zeros((total_sample_size, 2))
        inputs = torch.zeros((total_sample_size, 2))

        for i in range(total_sample_size):
            x, z, u = model()
            ground_truth[i, :] = x
            measurements[i, :] = z
            inputs[i, :] = u

        return ground_truth[starting_point*sample_length:], \
               measurements[starting_point*sample_length:], \
               inputs[starting_point*sample_length:]
