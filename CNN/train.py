import torch
import time
from tqdm import tqdm


def train_model(
    model, loss_fn, optimizer, early_stopping, epochs, train_loader, val_loader, device
):
    """
    Train the model
    Arguments:
        model: a pytorch model
        loss_fn: loss function for the model
        optimizer: the optimizer
        early_stopping: an early stopping class
        epochs: integer, number of epochs for the training
        train_loader: pytorch dataLoader with the training data
        val_loader: pytorch dataLoader with the validation data
        device: device in which training is executed

    Returns:
        model: the trained model
        train_acc: float representing training accuracy
        val_acc: float representing validation accuracy
    """

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # Reset metrics
        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0
        val_correct = 0

        # Training loop
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Training steps
                optimizer.zero_grad()  # zero the gradients
                output = model(inputs)  # make predictions for the batch
                loss = loss_fn(output, targets)  # compute loss and gradients
                loss.backward()
                optimizer.step()  # adjust learning weights
                # loss for the complete epoch more accurate. Depending on the
                # batch size and the length of dataset, last batch might be smaller than
                # the specified batch size. If that’s the case,
                # multiply the loss by the current batch size
                train_loss += loss.item() * inputs.size(0)  # update training loss
                train_correct += (output == targets).sum().item()  # update tr. accuracy

        # Validation loop
        model.eval()
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Validation steps
            with torch.no_grad():  # not calculating gradients every step
                output = model(inputs)  # forward pass
                loss = loss_fn(output, targets)  # calculate loss
                val_loss += loss.item() * inputs.size(0)  # update validation loss
                val_correct += (output == targets).sum().item()  # update val accuracy

        # Calculate average losses and accuracy
        train_loss = train_loss / len(train_loader.sampler)
        val_loss = val_loss / len(val_loader.sampler)
        train_acc = (train_correct / len(train_loader.sampler)) * 100
        val_acc = (val_correct / len(val_loader.sampler)) * 100

        # Display metrics at the end of each epoch
        print(
            f"""Epoch: {epoch} \tTraining Loss: {round(train_loss,5)} \t
                Validation Loss: {round(val_loss,5)} \tTime taken: {round(time.time() - start_time, 3)}s"""  # noqa: E501
        )

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    return model, train_acc, val_acc
