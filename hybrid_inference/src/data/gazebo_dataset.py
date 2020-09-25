import torch
from torch.utils.data import Dataset


class GazeboDataset(Dataset):

    def __init__(self, num_samples: int, dir_path: str, sample_length: int):
        """
        Initialises the dataset. Reads all the samples into memory so be careful with large datasets.
        :param num_samples: int The total number of samples that are saved in the directory. Equivelent to number of timesteps recorded.
        :param dir_path: str The path to the directory containing the samples. Can be absolute or relative.
        :param sample_length: int The number of timesteps in a sample for training.
        """
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.dir_path = dir_path
        # read in all samples THIS WILL NEED TO BE DIFFERENT IF THERE IS A REALLY LARGE SET.
        self.ys = list()
        self.Fs = list()
        self.ground_truths = list()
        for i in range(self.num_samples - self.sample_length):
            # read in ys
            y = torch.jit.load(self.dir_path + "y-" + str(i) + ".pt").named_parameters()
            next(y)
            next(y)  # this is for recording long 1 only.
            self.ys.append(next(y)[1])

            # read in Fs
            F = torch.jit.load(self.dir_path + "F-" + str(i) + ".pt").named_parameters()
            next(F)
            next(F)  # this is for recording long 1 only
            self.Fs.append(next(F)[1])

            # read in ground truths
            g_t = torch.load(self.dir_path + "odom-" + str(i) + ".pt")
            self.ground_truths.append(g_t[1])

    def __len__(self):
        """
        Returns the total number of samples taking into account the length of each sample
        :return: int number of samples available
        """
        return len(self.ys)

    def total_samples(self):
        return len(self.ys) * self.sample_length

    def __getitem__(self, index):
        """
        Fetches a single sample.
        :param index: The index of the sample.
        :return: (tensor, tensor, tensor) A triple of ys, Fs, ground truths.
        """
        return torch.stack(self.ys[index:index + self.sample_length]), \
               torch.stack(self.Fs[index:index + self.sample_length]), \
               torch.stack(self.ground_truths[index:index + self.sample_length])


class GazeboDatasetH(Dataset):

    def __init__(self, num_samples: int, dir_path: str, sample_length: int):
        """
        Initialises the dataset. Reads all the samples into memory so be careful with large datasets.
        :param num_samples: int The total number of samples that are saved in the directory. Equivelent to number of timesteps recorded.
        :param dir_path: str The path to the directory containing the samples. Can be absolute or relative.
        :param sample_length: int The number of timesteps in a sample for training.
        """
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.dir_path = dir_path
        # read in all samples THIS WILL NEED TO BE DIFFERENT IF THERE IS A REALLY LARGE SET.
        self.ys = list()
        self.Fs = list()
        self.Hs = list()
        self.ground_truths = list()
        for i in range(self.num_samples - self.sample_length):
            # read in ys
            y = torch.jit.load(self.dir_path + "y-" + str(i) + ".pt").named_parameters()
            next(y)
            self.ys.append(next(y)[1])

            # read in Fs
            F = torch.jit.load(self.dir_path + "F-" + str(i) + ".pt").named_parameters()
            next(F)
            self.Fs.append(next(F)[1])

            H = torch.jit.load(self.dir_path + "H-" + str(i) + ".pt").named_parameters()
            next(H)
            self.Hs.append(next(H)[1])

            # read in ground truths
            g_t = torch.load(self.dir_path + "odom-" + str(i) + ".pt")
            self.ground_truths.append(g_t[1])

    def __len__(self):
        """
        Returns the total number of samples taking into account the length of each sample
        :return: int number of samples available
        """
        return len(self.ys)

    def total_samples(self):
        return len(self.ys) * self.sample_length

    def __getitem__(self, index):
        """
        Fetches a single sample.
        :param index: The index of the sample.
        :return: (tensor, tensor, tensor) A triple of ys, Fs, ground truths.
        """
        return torch.stack(self.ys[index:index + self.sample_length]), \
               torch.stack(self.Fs[index:index + self.sample_length]), \
               torch.stack(self.Hs[index:index + self.sample_length]), \
               torch.stack(self.ground_truths[index:index + self.sample_length])
