"""Utility functions for training neural networks."""

import time

import torch

from scalai.dpnn.utils_eval import compute_accuracy, get_right_ddp


def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logging_interval: int = 50,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> tuple[list[float], list[float], list[float]]:
    """Train your model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    num_epochs : int
        The number of epochs to train
    train_loader : torch.utils.data.DataLoader
        The training dataloader.
    valid_loader : torch.utils.data.DataLoader
        The validation dataloader.
    test_loader : torch.utils.data.DataLoader
        The testing dataloader.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    device : torch.device
        The device to train on.
    logging_interval : int
        The logging interval.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        An optional learning rate scheduler.

    Returns
    -------
    List[float]
        The loss history.
    List[float]
        The training accuracy history.
    List[float]
        The validation accuracy history.
    """
    ## start = ... # Start timer to measure training time.
    start = time.perf_counter()

    # Initialize history lists for loss, training accuracy, and validation accuracy.
    loss_history, train_acc_history, valid_acc_history = [], [], []

    # ACTUAL TRAINING STARTS HERE.
    for epoch in range(num_epochs):  # Loop over epochs.
        # IMPLEMENT TRAINING LOOP HERE.
        #
        ## Set model to training mode.
        #  Thus, layers like dropout which behave differently on train and
        #  test procedures know what is going on and can behave accordingly.
        model.train()

        for batch_idx, (features, targets) in enumerate(
            train_loader
        ):  # Loop over mini batches.
            # CONVERT DATASET TO USED DEVICE.
            features = features.to(device)
            targets = targets.to(device)
            #
            # FORWARD & BACKWARD PASS
            ## logits = ...  # Get predictions of model with current parameters.
            logits = model(features)
            ## loss = ...  # Calculate cross-entropy loss on current mini-batch.
            loss = torch.nn.functional.cross_entropy(logits, targets)
            ## Zero out gradients.
            optimizer.zero_grad()
            ## Calculate gradients of loss w.r.t. model parameters in backward pass.
            loss.backward()
            ## Perform single optimization step to update model parameters via optimizer.
            optimizer.step()
            #
            # LOGGING
            ## Append loss to history list.
            loss_history.append(loss.item())

            if not batch_idx % logging_interval:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                    f"| Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| Loss: {loss:.4f}"
                )

        # VALIDATION STARTS HERE.
        #
        ## Set model to evaluation mode.
        model.eval()

        with (
            torch.no_grad()
        ):  # Disable gradient calculation to reduce memory consumption.
            # COMPUTE ACCURACY OF CURRENT MODEL PREDICTIONS ON TRAINING + VALIDATION DATASETS.
            train_acc = compute_accuracy(
                model, train_loader, device=device
            )  # Compute accuracy on training data.
            valid_acc = compute_accuracy(
                model, valid_loader, device=device
            )  # Compute accuracy on validation data.

            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                f"| Train: {train_acc :.2f}% "
                f"| Validation: {valid_acc :.2f}%"
            )

            ## APPEND ACCURACY VALUES TO CORRESPONDING HISTORY LISTS.
            train_acc_history.append(train_acc)
            valid_acc_history.append(valid_acc)

        # Stop timer and calculate training time elapsed after epoch.
        elapsed = time.perf_counter() - start

        print(f"Elapsed time: {elapsed / 60:.2f} min")

        if scheduler is not None:  # Adapt learning rate.
            original_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            new_lr = scheduler.get_last_lr()[0]
            if original_lr != new_lr:
                print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}: LR updated to {new_lr}")

    elapsed = time.perf_counter() - start
    ## Print overall training time.
    print(f"Total Training Time: {elapsed / 60:.2f} min")

    # FINAL TESTING STARTS HERE.
    #
    test_acc = compute_accuracy(
        model, test_loader, device
    )  # Compute accuracy on test data.
    ## Print test accuracy.
    print(f"Test accuracy: {test_acc:.2f}%")

    ## Return history lists for loss, training accuracy, and validation accuracy.
    return loss_history, train_acc_history, valid_acc_history


def train_model_ddp(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[float], list[float], list[float]]:
    """Train the model in distributed data-parallel fashion.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    num_epochs : int
        The number of epochs to train.
    train_loader : torch.utils.data.DataLoader
        The training dataloader.
    valid_loader : torch.utils.data.DataLoader
        The validation dataloader.
    optimizer : torch.optim.Optimizer
        The optimizer to use.

    Returns
    -------
    List[float]
        The epoch-wise loss history.
    List[float]
        The epoch-wise training accuracy history.
    List[float]
        The epoch-wise validation accuracy history.
    """
    start = time.perf_counter()  # Start timer to measure training time.
    rank = torch.distributed.get_rank()  # Get local process ID (= rank).
    world_size = torch.distributed.get_world_size()  # Get overall number of processes.

    loss_history, train_acc_history, valid_acc_history = (
        [],
        [],
        [],
    )  # Initialize history lists.

    # Actual training starts here.
    for epoch in range(num_epochs):  # Loop over epochs.
        train_loader.sampler.set_epoch(
            epoch
        )  # Set current epoch for distributed dataloader.
        ## Set model to training mode.
        model.train()

        for batch_idx, (features, targets) in enumerate(
            train_loader
        ):  # Loop over mini batches.
            # Convert dataset to GPU device.
            features = features.cuda()
            targets = targets.cuda()

            # FORWARD & BACKWARD PASS
            ## logits = ...  # Get predictions of current model from forward pass.
            logits = model(features)
            ## loss = ...  # Use cross-entropy loss.
            loss = torch.nn.functional.cross_entropy(logits, targets)
            ## Zero out gradients (by default, gradients are accumulated in buffers in backward pass).
            optimizer.zero_grad()
            ## Backward pass.
            loss.backward()
            ## Update model parameters in single optimization step.
            optimizer.step()
            #
            # LOGGING
            ## Calculate effective mini-batch loss as process-averaged mini-mini-batch loss.
            ## Sum up mini-mini-batch losses from all processes and divide by number of processes.
            ## Use collective communication functions from `torch.distributed` package.
            # Note that `torch.distributed` collective communication functions will only
            # work with `torch` tensors, i.e., floats, ints, etc. must be converted before!
            ## Append globally averaged loss of this epoch to history list.

            # sum losses from all processes
            torch.distributed.all_reduce(loss)
            loss /= world_size
            loss_history.append(loss.item())

            if rank == 0:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                    f"| Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| Averaged Loss: {loss:.4f}"
                )

        # Validation starts here.

        ## Set model to evaluation mode.
        model.eval()

        with torch.no_grad():  # Disable gradient calculation.
            # Validate model in data-parallel fashion.
            # Determine number of correctly classified samples and overall number
            # of samples in training and validation dataset.
            #
            right_train, num_train = get_right_ddp(model, train_loader)
            right_valid, num_valid = get_right_ddp(model, valid_loader)
            #
            ## Sum up number of correctly classified samples in training dataset,
            ## overall number of considered samples in training dataset,
            ## number of correctly classified samples in validation dataset,
            ## and overall number of samples in validation dataset over all processes.
            ## Use collective communication functions from `torch.distributed` package.
            torch.distributed.all_reduce(right_train)
            torch.distributed.all_reduce(right_valid)
            torch.distributed.all_reduce(num_train)
            torch.distributed.all_reduce(num_valid)
            #
            # Note that `torch.distributed` collective communication functions will only
            # work with torch tensors, i.e., floats, ints, etc. must be converted before!
            # From these values, calculate overall training + validation accuracy.
            #
            train_acc = right_train.item() / num_train.item() * 100
            valid_acc = right_valid.item() / num_valid.item() * 100
            ## Append accuracy values to corresponding history lists.
            train_acc_history.append(train_acc)
            valid_acc_history.append(valid_acc)

            if rank == 0:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                    f"| Train: {train_acc :.2f}% "
                    f"| Validation: {valid_acc :.2f}%"
                )

        elapsed = (time.perf_counter() - start) / 60  # Measure training time per epoch.
        elapsed = torch.Tensor([elapsed]).cuda()
        torch.distributed.all_reduce(elapsed)
        elapsed /= world_size
        if rank == 0:
            print(f"Time elapsed: {elapsed.item()} min")

    elapsed = (
        time.perf_counter() - start
    )  # Stop timer and calculate training time elapsed after epoch.
    elapsed = torch.Tensor([elapsed / 60]).cuda()
    ## Calculate average training time elapsed after each epoch over all processes,
    ## i.e., sum up times from all processes and divide by overall number of processes.
    ## Use collective communication functions from torch.distributed package.
    # Note that torch.distributed collective communication functions will only
    # work with torch tensors, i.e., floats, ints, etc. must be converted before!
    torch.distributed.all_reduce(elapsed)
    elapsed /= world_size

    if rank == 0:
        ## Print process-averaged training time after each epoch.
        print("Total Training Time:", elapsed.item(), "min")
        torch.save(loss_history, f"loss_{world_size}_gpu.pt")
        torch.save(train_acc_history, f"train_acc_{world_size}_gpu.pt")
        torch.save(valid_acc_history, f"valid_acc_{world_size}_gpu.pt")

    return loss_history, train_acc_history, valid_acc_history
