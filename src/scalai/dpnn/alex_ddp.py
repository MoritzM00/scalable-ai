"""Distributed data-parallel training of AlexNet on the CIFAR-10 dataset."""

import os

import torch
import torch.distributed as dist
import torchvision
from model import AlexNet
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_data import get_dataloaders_cifar10_ddp, get_transforms_cifar10
from utils_eval import compute_accuracy_ddp
from utils_train import train_model_ddp


def main():
    """Distributed data-parallel training of AlexNet on the CIFAR-10 dataset."""
    world_size = int(
        os.getenv("SLURM_NPROCS")
    )  # Get overall number of processes from SLURM environment variable.
    rank = int(
        os.getenv("SLURM_PROCID")
    )  # Get individual process ID from SLURM environment variable.
    print(
        f"Rank, world size, device count: {rank}, {world_size}, {torch.cuda.device_count()}"
    )

    if rank == 0:
        if dist.is_available():
            print(
                "Distributed package available...[OK]"
            )  # Check if distributed package available.
        if dist.is_nccl_available():
            print("NCCL backend available...[OK]")  # Check if NCCL backend available.

    # On each host with N GPUs, spawn up N processes, while ensuring that
    # each process individually works on a single GPU from 0 to N-1.

    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
    port = "29500"
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port

    # Initialize DDP.
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    ## Check if process group has been initialized successfully.
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    if dist.is_initialized():
        print("Process group initialized successfully...[OK]")  # Check initialization

    ## Check used backend.
    print(dist.get_backend(), "backend used...[OK]")

    b = 256  # Set batch size.
    e = 100  # Set number of epochs to be trained.
    data_root = "~/scalable-ai/impl/sheet_5/data/"  # Path to data dir

    # Get transforms for data preprocessing to make smaller CIFAR-10 images work with AlexNet using helper function from task 1.
    train_transforms, test_transforms = get_transforms_cifar10()

    # Get distributed dataloaders for training and validation data on all ranks.
    train_loader, valid_loader = get_dataloaders_cifar10_ddp(
        batch_size=b,
        data_root=data_root,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Get device used for training.
    model = AlexNet(10).to(
        device=device
    )  # Create AlexNet model with 10 classes for CIFAR-10 and move it to GPU.
    ddp_model = DDP(model)  # Wrap model with DDP.

    # Set up stochastic gradient descent optimizer from torch.optim package.
    # Use a momentum of 0.9 and a learning rate of 0.1.
    # Use parameters of DDP model here!
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)

    # Train DDP model.
    train_model_ddp(
        model=ddp_model,
        optimizer=optimizer,
        num_epochs=e,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    # Test final model on root.
    if dist.get_rank() == 0:
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transforms,
        )  # Get dataset for test data.
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=b, shuffle=False
        )  # Get dataloader for test data.
        test_acc = compute_accuracy_ddp(
            model, test_loader
        )  # Compute accuracy on test data.
        ## Print test accuracy.
        print(f"Test accuracy: {test_acc:.2f}")

    ## Destroy process group.
    dist.destroy_process_group()


# MAIN STARTS HERE.
if __name__ == "__main__":
    main()
