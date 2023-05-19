import os
import torch
import random
import time
import dataset_loader
import pre_processing
import traceback
import DNN_model
import torch.nn as nn
import rule_based_filtering as filter
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer


print(torch.__version__)

"""
NETWORK INFO

    Rule Based:
        - valid ID
        - time interval
        - valid DLC

    Deep neural network
        Features:
            - message ID
            - hamming distance
            - entropy of data field
            - bytes of importance
        Structure:
            - Feed forward
            - 5 hidden layers [100, 100, 80, 60, 40]
            - binary output (1 positive, 0 negative)
            - ReLu activation for hidden layers
            - sigmoid for output layer
            - Adam optimizer
            - binary cross-entropy as loss function 

"""

############################################################
####               MODEL GENERAL SETTINGS               ####
############################################################

# Set the model directory
cwd = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(cwd, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set random seed for reproducibility
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device used: {}".format(device))

# model settings
input_dim = 5
output_dim = 1
# n = 40  # number of inputs
batch_size = 128

epochs = 50
early_stopping_patience = 5
early_stopping_min_delta = 0
lr = 0.001

# Early stopping
# credits to Massaro
# TODO: check if possible the copy otherwise re-implement it


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


############################################################
####                   PRE-PROCESSING                   ####
############################################################

dataset = dataset_loader.get_dataset()

# initialize only if clean dataset with no attacks
# TODO: initialize rules always on, check and save rules for IDs only if already saved
#       or if the flag indicates a malicious attack
# filter.initialize_rules(dataset)

whitelisted_dataset, id_blacklist, period_blacklist, dlc_blacklist = filter.filter(
    dataset
)  # noqa: E501

b1 = 0
b2 = 1
train_size = 0.7
val_size = 0.2
test_size = 0.1
features = pre_processing.get_features(whitelisted_dataset, b1, b2)
# train = 80%, val = 20%
train, val = train_test_split(features, test_size=val_size)
# train = 70%, val = 20%, test = 10%
train, test = train_test_split(train, test_size=test_size / (train_size + test_size))
print(f"Training dataset length: {len(train)}")
print(f"Validation dataset length: {len(val)}")
print(f"Test dataset length: {len(test)}")
train_loader = dataset_loader.get_data_loader(train, batch_size=batch_size)
val_loader = dataset_loader.get_data_loader(val, batch_size=batch_size)
test_loader = dataset_loader.get_data_loader(test, batch_size=1)

############################################################
####               BUILD AND TRAIN MODEL                ####
############################################################

# General settings
n_features = 5  # features.shape() boh
input_shape = [1, n_features]
print("Input shape: {}".format(input_shape))

model = DNN_model.DNN(input_shape)
model = model.to(device)
model.eval()

# Definition of training settings
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
early_stopping = EarlyStopping(
    patience=early_stopping_patience, min_delta=early_stopping_min_delta
)

print("############################################################")
print("####                 TRAINING THE MODEL                 ####")
print("############################################################")

# Training loop
try:
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
                train_loss += loss.item() * input.size(0)  # update training loss
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
                val_loss += loss.item() * input.size(0)  # update validation loss
                val_correct += (output == targets).sum().item()  # update val accuracy

        # Calculate average losses and accuracy
        train_loss = train_loss / len(train_loader.sampler)
        val_loss = val_loss / len(val_loader.sampler)
        train_acc = (train_correct / len(train_loader.sampler)) * 100
        val_acc = (val_correct / len(val_loader.sampler)) * 100

        # Display metrics at the end of each epoch
        print(
            f"""Epoch: {epoch} \tTraining Loss: {train_loss} \t
            Validation Loss: {val_loss} \tTime taken: {start_time - time.time()}"""
        )

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    print("Saving model...")

    # Model saving routine
    model_dir = os.path.join(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

except Exception as e:
    traceback.print_exc()


