import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .gazebo_dataset import GazeboDataset, GazeboDatasetH


def get_dataloaders(dataset_path: str, dataset_len: int, val: float, test: float, sample_len: int = 10,
                    batch_size: int = 1, shuffle=True, seed: int = 42, extras={}, H=False):
    # Get create a ChestXrayDataset object
    if H:
        dataset = GazeboDatasetH(dataset_len, dataset_path, sample_len)
    else:
        dataset = GazeboDataset(dataset_len, dataset_path, sample_len)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # Create the validation split from the full dataset
    val_split = int(np.floor(val * dataset_size))
    train_ind, val_ind = all_indices[val_split:], all_indices[: val_split]

    # Separate a test split from the training dataset
    test_split = int(np.floor(test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split:], train_ind[: test_split]

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                             pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)

    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)
