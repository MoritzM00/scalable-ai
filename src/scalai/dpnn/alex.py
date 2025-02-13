"""Serial script for training AlexNet on CIFAR-10 dataset."""

import torch

from scalai.dpnn.model import AlexNet
from scalai.dpnn.utils_data import get_dataloaders_cifar10, get_transforms_cifar10
from scalai.dpnn.utils_train import train_model


def main():
    """Train AlexNet on CIFAR-10 dataset on a single device."""
    # Transforms on your data allow you to take it from its source state and transform it into ready-for-training data.
    # Get transforms applied to CIFAR-10 data for training and inference.
    train_transforms, test_transforms = get_transforms_cifar10()

    b = 256  # Set mini-batch size hyperparameter.
    data_root = "~/scalable-ai/impl/sheet_5/data/"  # Path to data dir.
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        data_root=data_root,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        seed=seed,
    )

    # Check loaded dataset.
    for images, labels in train_loader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        print("Class labels of 10 examples:", labels[:10])
        break

    # SETTINGS
    # Include into your main script to be executed when running as a batch job later on.

    e = 100  # Number of epochs
    lr = 0.1

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Print used device.
    print(f"Device: {device}")

    model = AlexNet(num_classes=10).to(device)
    # Build an instance of AlexNet with 10 classes for CIFAR-10 and convert it to the used device.
    ## Print model.
    print(model)

    # Set up an SGD optimizer from the `torch.optim` package.
    # Use a momentum of 0.9 and a learning rate of 0.1.
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Set up a LR scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, mode="max"
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # TRAIN MODEL.
    loss_history, train_acc_history, valid_acc_history = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=e,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        logging_interval=100,
    )

    # Save history lists for loss, training accuracy, and validation accuracy.S
    torch.save(loss_history, "loss.pt")
    torch.save(train_acc_history, "train_acc.pt")
    torch.save(valid_acc_history, "valid_acc.pt")


if __name__ == "__main__":
    main()
