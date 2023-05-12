import os
import torch
import random
import dataset_loader
import pre_processing
import rule_based_filtering as filter
import numpy as np
import pandas as pd
from pathlib import Path

print(torch.__version__)

'''
NETWORK INFO


'''

############################################################
####               MODEL GENERAL SETTINGS               ####
############################################################

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
n = 40  # number of inputs
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
        if self.best_loss == None:
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
# filter.initialize_rules(dataset)

whitelisted_dataset, id_blacklist, period_blacklist, dlc_blacklist = filter.filter(dataset)  # noqa: E501

b1 = 0
b2 = 1
features = pre_processing.get_features(whitelisted_dataset,b1, b2)

data_loader = dataset_loader.get_data_loader(features, batch_size)


# TRAINING THE MODEL





