from .synthetic_position_dataset import SyntheticPositionDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


def get_dataloaders(train_samples: int = 1000, val_samples: int = 200, test_samples: int = 500,
                    sample_length: int = 10, starting_point: int = 0, extras={}):
    torch.manual_seed(42)
    x0 = torch.randn(4)
    train_seed = 42
    val_seed = 3
    test_seed = 11
    train_dataset = SyntheticPositionDataset(x0=x0, n_samples=train_samples, sample_length=sample_length,
                                             starting_point=starting_point, seed=train_seed)

    val_dataset = SyntheticPositionDataset(x0=x0, n_samples=val_samples, sample_length=sample_length,
                                             starting_point=starting_point, seed=val_seed)

    test_dataset = SyntheticPositionDataset(x0=x0, n_samples=test_samples, sample_length=sample_length,
                                             starting_point=starting_point, seed=test_seed)

    #### This section is taken from code provided with thanks by Jenny Hamer.
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
