"""Utility functions for data handling in the context of DPNNs."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision


def get_transforms_cifar10() -> (
    tuple[torchvision.transforms.Compose, torchvision.transforms.Compose]
):
    """Get transforms applied to CIFAR-10 data for AlexNet training and inference.

    Returns
    -------
    torchvision.transforms.Compose
        The transforms applied to CIFAR-10 for training AlexNet.
    torchvision.transforms.Compose
        The transforms applied to CIFAR-10 to run inference with AlexNet.
    """
    # Transforms applied to training data (randomness to make network more robust against overfitting)
    train_transforms = (
        torchvision.transforms.Compose(  # Compose several transforms together.
            [
                torchvision.transforms.Resize(
                    (70, 70)
                ),  # Upsample CIFAR-10 images to make them work with AlexNet.
                torchvision.transforms.RandomCrop(
                    (64, 64)
                ),  # Randomly crop image to make NN more robust against overfitting.
                torchvision.transforms.ToTensor(),  # Convert image into torch tensor.
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1] via (image-mean)/std.
            ]
        )
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.CenterCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return train_transforms, test_transforms


def make_train_validation_split(
    train_dataset: torchvision.datasets.CIFAR10,
    seed: int = 123,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split original CIFAR-10 training data into train and validation sets.

    Parameters
    ----------
    train_dataset : torchvision.datasets.CIFAR10
        The original CIFAR-10 training dataset.
    seed : int
        The seed used to split the data.
    validation_fraction : float
        The fraction of samples used for validation.

    Returns
    -------
    numpy.ndarray
        The sample indices for the training dataset.
    numpy.ndarray
        The sample indices for the validation dataset.
    """
    num_samples = len(
        train_dataset
    )  # Get overall number of samples in original training data.
    rng = np.random.default_rng(
        seed=seed
    )  # Set same seed over all ranks for consistent train-test split.
    idx = np.arange(0, num_samples)  # Construct array of all indices.
    rng.shuffle(idx)  # Shuffle them.
    num_validate = int(
        validation_fraction * num_samples
    )  # Determine number of validation samples from validation split.
    return (
        idx[num_validate:],
        idx[0:num_validate],
    )  # Extract and return train and validation indices.


def get_dataloaders_cifar10(
    batch_size: int,
    data_root: str = "data",
    validation_fraction: float = 0.1,
    train_transforms: Callable[[Any], Any] = None,
    test_transforms: Callable[[Any], Any] = None,
    seed: int = 123,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Get dataloaders for training, validation, and testing on the CIFAR-10 dataset.

    Parameters
    ----------
    batch_size : int
        The mini-batch size.
    data_root : str
        The path to the dataset.
    validation_fraction : float
        The fraction of the original training data used for validation.
    train_transforms : Callable[[Any], Any]
        The transform applied to the training data.
    test_transforms : Callable[[Any], Any]
        The transform applied to the validation/testing data (inference).
    seed : int
        The seed for the validation-train split.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    torch.utils.data.DataLoader
        The testing dataloader.
    """
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=train_transforms, download=True
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=test_transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    train_indices, valid_indices = make_train_validation_split(
        train_dataset, seed, validation_fraction
    )  # Get train and validation indices.

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10_ddp(
    batch_size: int,
    data_root: str = "data",
    validation_fraction: float = 0.1,
    train_transforms: Callable[[Any], Any] = None,
    test_transforms: Callable[[Any], Any] = None,
    seed=123,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get distributed CIFAR-10 dataloaders for training and validation in a DDP setting.

    Parameters
    ----------
    batch_size : int
        The batch size.
    data_root : str
        The path to the data directory.
    validation_fraction : float
        The fraction of training samples used for validation.
    train_transforms : Callable[[Any], Any]
        The transform applied to the training data.
    test_transforms : Callable[[Any], Any]
        The transform applied to the testing data (inference).
    seed : int
        Seed for train-validation split.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    """
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()
    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    if (
        dist.get_rank() == 0
    ):  # Only root shall download dataset if data is not already there.
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, transform=train_transforms, download=True
        )

    dist.barrier(device_ids=[torch.cuda.current_device()])  # Barrier

    if (
        dist.get_rank() != 0
    ):  # Other ranks must not download dataset at the same time in parallel.
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, transform=train_transforms
        )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, transform=test_transforms
    )

    ## PERFORM INDEX-BASED TRAIN-VALIDATION SPLIT OF ORIGINAL TRAINING DATA.
    ## train_indices, valid_indices = ...  # Extract train and validation indices using helper function from task 1.
    train_indices, valid_indices = make_train_validation_split(
        train_dataset, seed, validation_fraction
    )

    # Split into training and validation dataset according to specified validation fraction.
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    # Sampler that restricts data loading to a subset of the dataset.
    # Especially useful in conjunction with DistributedDataParallel.
    # Each process can pass a DistributedSampler instance as a DataLoader sampler,
    # and load a subset of the original dataset that is exclusive to it.

    # Get samplers.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    # Get dataloaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=valid_sampler,
    )

    return train_loader, valid_loader
