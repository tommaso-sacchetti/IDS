import os
import random
import torch
import dataset_loader
import pre_processing
import traceback
import DNN_model
import torch.nn as nn
import global_variables as glob
import rule_based_filtering as filter
import numpy as np
import torch.optim as optim
from train import train_model
from sklearn.model_selection import train_test_split

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
############################################################
####               MODEL GENERAL SETTINGS               ####
############################################################
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
batch_size = 128
epochs = 50
early_stopping_patience = 5
early_stopping_min_delta = 0
lr = 0.001


class EarlyStopping:
    # Early stopping
    # credits to Massaro
    # TODO: check if possible the copy otherwise re-implement it

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
############################################################
####                   PRE-PROCESSING                   ####
############################################################
############################################################

dataset = dataset_loader.get_dataset()

#############
# TO REMOVE #
#############
dataset = dataset.drop(dataset.index[1000:])

filter.initialize_rules(dataset)
whitelisted_dataset, id_blacklist, period_blacklist, dlc_blacklist = filter.filter(
    dataset
)  # noqa: E501

# TODO: IMPORTANCE BYTES, FIND THE RIGHT ONES
b1 = glob.b1
b2 = glob.b2

# Split train, validation and test
train_size = 0.8
val_size = 0.2
#test_size = 0.1
features = pre_processing.get_features(whitelisted_dataset, b1, b2)
train, val = train_test_split(features, test_size=val_size)
#train, test = train_test_split(train, test_size=test_size / (train_size + test_size))
print(f"Training dataset length: {len(train)}")
print(f"Validation dataset length: {len(val)}")
#print(f"Test dataset length: {len(test)}")

train_loader = dataset_loader.get_data_loader(train, batch_size=batch_size)
val_loader = dataset_loader.get_data_loader(val, batch_size=batch_size)
#test_loader = dataset_loader.get_data_loader(test, batch_size=1)



############################################################
############################################################
####               BUILD AND TRAIN MODEL                ####
############################################################
############################################################

model = DNN_model.DNN(input_dim)
model = model.to(device)
model.eval()
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
    model, train_acc, val_acc = train_model(
        model,
        loss_fn,
        optimizer,
        early_stopping,
        epochs,
        train_loader,
        val_loader,
        device
    )

    print("Saving model...")
    # Model saving routine
    model_dir = os.path.join(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

except Exception as e:
    traceback.print_exc()
    print(e)
