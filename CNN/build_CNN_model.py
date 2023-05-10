import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import logging

print(torch.__version__)

"""
General:
    dataset: 80% training, 20% testing
    dropout of 0.1 before max pooling
    200 epochs

Binary classification:
    Adam optimizer
    lr 0.0001
    binary_crossentropy
    tanh input activation function
    sigmoid output activation function
    single convolutional layer with 512 filter

Multiclass classification
    Nadam optimizer
    lr 0.0001
    categorical_crossentropy
    sigmoid input activation funtion
    softmax output activation function
    two convolutional layers with 512 filter
"""

# Set random seed for reproducibility
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device used: {}".format(device))

############################################################
####               MODEL GENERAL SETTINGS               ####
############################################################

input_dim = 5
output_dim = 1

n = 40  # number of inputs
batch_size = 256
epochs = 200
early_stopping_patience = 5
early_stopping_min_delta = 0
lr = 0.0001
dropout = 0.1

# Early stopping: credits to Massaro
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
