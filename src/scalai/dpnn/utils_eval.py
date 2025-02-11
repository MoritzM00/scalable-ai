"""Utility functions for evaluation of models."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

# EVALUATION
# Save to a separate Python module file `utils_eval.py` to import the functions from
# into your main script and run the training as a batch job later on.
# Add imports as needed.


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute the accuracy of the model's predictions on given labeled data.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_loader : torch.utils.data.DataLoader
        The dataloader.
    device : torch.device
        The device to use.

    Returns
    -------
    float
        The model's accuracy on the given dataset in percent.
    """
    with torch.no_grad():  # Disable gradient calculation to reduce memory consumption.
        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_examples = (
            0,
            0,
        )  # Initialize number of correctly predicted and overall samples, respectively.

        for i, (features, targets) in enumerate(data_loader):
            # CONVERT DATASET TO USED DEVICE.
            features = features.to(device)
            targets = targets.to(device)

            #
            # CALCULATE PREDICTIONS OF CURRENT MODEL ON FEATURES OF INPUT DATA.
            logits = model(features)
            ## Determine class with highest score.
            pred = torch.argmax(logits, dim=1)
            ## Compare predictions to actual labels to determine number of correctly predicted samples.
            correct_pred += torch.sum(pred == targets)
            ## Determine overall number of samples.
            num_examples += targets.size(0)

    # CALCULATE AND RETURN ACCURACY AS PERCENTAGE OF CORRECTLY PREDICTED SAMPLES.
    return correct_pred.float() / num_examples * 100


def plot_results(res_path: pathlib.Path | str) -> None:
    """
    Plot training loss and training and validation accuracy.

    Parameters
    ----------
    res_path : pathlib.Path
        The path to the results pickle files.
    """
    label_size = 16
    res_path = pathlib.Path(res_path)
    train_loss = np.array(torch.load(res_path / pathlib.Path("loss.pt")))
    train_acc = np.array(torch.load(res_path / pathlib.Path("train_acc.pt")))
    valid_add = np.array(torch.load(res_path / pathlib.Path("valid_acc.pt")))
    n_epochs = len(train_acc)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    epochs_loss = np.linspace(1, n_epochs, train_loss.shape[0])
    epochs_acc = np.linspace(1, n_epochs, n_epochs)
    ax1.plot(epochs_loss, train_loss, lw=0.1, color="k")
    ax1.grid()
    ax1.set_xlabel("Epoch", weight="bold", fontsize=label_size)
    ax1.set_ylabel("Train loss", weight="bold", fontsize=label_size)
    ax2.plot(epochs_acc, train_acc, label="Training", lw=2, color="k")
    ax2.plot(
        epochs_acc, valid_add, label="Validation", lw=2, color=(0.1, 0.6294, 0.5588)
    )
    ax2.set_xlabel("Epoch", weight="bold", fontsize=label_size)
    ax2.set_ylabel("Accuracy / %", weight="bold", fontsize=label_size)
    ax2.legend(fontsize=label_size)
    ax2.grid()
    plt.tight_layout()
    plt.savefig(res_path / pathlib.Path("results.pdf"))
    plt.show()


if __name__ == "__main__":
    plot_results(res_path="../res/gpu_1")
    plot_results(res_path="../res/gpu_4")
    plot_results(res_path="../res/gpu_16")


def get_right_ddp(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the number of correctly predicted samples and the overall number of samples in a given dataset.

    This function is needed to compute the accuracy over multiple processors in a distributed data-parallel setting.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_loader : torch.utils.data.DataLoader
        The dataloader.

    Returns
    -------
    torch.Tensor
        The number of correctly predicted samples.
    torch.Tensor
        The overall number of samples in the dataset.
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.float().cuda()
            # CALCULATE PREDICTIONS OF CURRENT MODEL ON FEATURES OF INPUT DATA.
            logits = model(features)
            ## Determine class with highest score.
            _, predicted_labels = torch.max(logits, 1)  # Get class with highest score.
            ## Update overall number of samples.
            num_examples += targets.size(0)
            ## Compare predictions to actual labels to determine number of correctly predicted samples.
            correct_pred += (predicted_labels == targets).sum().item()

    correct_pred = torch.Tensor([correct_pred]).cuda()
    num_examples = torch.Tensor([num_examples]).cuda()
    return correct_pred, num_examples


def compute_accuracy_ddp(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> float:
    """
    Compute the accuracy of the model's predictions on given labeled data.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    data_loader : torch.utils.data.DataLoader
        The dataloader.

    Returns
    -------
    float
        The model's accuracy on the given dataset in percent.
    """
    correct_pred, num_examples = get_right_ddp(model, data_loader)
    return correct_pred.item() / num_examples.item() * 100
