from .synthetic_position_dataset import SyntheticPositionDataset
from .synthetic_input_dataset import SyntheticInputDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


def get_dataloaders(train_samples: int = 1000, val_samples: int = 200, test_samples: int = 500,
                    sample_length: int = 10, starting_point: int = 0, batch_size:int = 1,
                    inputs=False, x0 = torch.tensor([0.0, 0.1, 0.0, 0.1]), extras={}):
    train_seed = 61
    val_seed = 51
    test_seed = 0
    if inputs:
        train_dataset = SyntheticInputDataset(x0=x0, n_samples=train_samples,
                                              sample_length=sample_length,
                                              starting_point=starting_point, seed=train_seed)

        val_dataset = SyntheticInputDataset(x0=x0, n_samples=val_samples,
                                            sample_length=sample_length,
                                            starting_point=starting_point, seed=val_seed)

        test_dataset = SyntheticInputDataset(x0=x0, n_samples=test_samples,
                                             sample_length=sample_length,
                                             starting_point=starting_point, seed=test_seed)
    else:
        train_dataset = SyntheticPositionDataset(x0=x0, n_samples=train_samples,
                                                 sample_length=sample_length,
                                                 starting_point=starting_point, seed=train_seed)

        val_dataset = SyntheticPositionDataset(x0=x0, n_samples=val_samples,
                                               sample_length=sample_length,
                                               starting_point=starting_point, seed=val_seed)

        test_dataset = SyntheticPositionDataset(x0=x0, n_samples=test_samples,
                                                sample_length=sample_length,
                                                starting_point=starting_point, seed=test_seed)

    #### This section is taken from code provided by Jenny Hamer, with thanks.
    # it allows easy specification of cuda options.
    num_workers = 0
    pin_memory = False

    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    ####

    train_sampler = RandomSampler(train_dataset, replacement=False)
    val_sampler = RandomSampler(val_dataset, replacement=False)
    test_sampler = RandomSampler(test_dataset, replacement=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
