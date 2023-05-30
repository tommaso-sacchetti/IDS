import os
import torch
import random
import traceback
import CNN_model
import dataset_loader
import pre_processing
import numpy as np
import torch.nn as nn
import torch.optim as optim
import global_variables as glob
from train import train_model
from sklearn.model_selection import train_test_split

print(torch.__version__)

"""
NETWORK INFO

General:
    dataset: 80% training, 20% testing
    dropout of 0.1 before max pooling
    200 epochs

Binary classification:
    Adam optimizer
    lr 0.0001
    loss: binary_crossentropy
    tanh input activation function
    sigmoid output activation function
    single convolutional layer with 512 filter

Multiclass classification
    Nadam optimizer
    lr 0.0001
    loss: categorical_crossentropy
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

# Set the model directory
cwd = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(cwd, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

input_dim = 9
output_dim = 1
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


############################################################
####                   PRE-PROCESSING                   ####
############################################################

dataset = dataset_loader.get_single_dataset(glob.attack_data_path)

# dataset = get_full_binary_dataset(train = True)
# test_dataset = get_full_binary_dataset(train = False)

# Split train, validation and test
train_size = 0.8
val_size = 0.2
features = pre_processing.get_binary_features(dataset)
train, val = train_test_split(features, test_size=val_size)
print(f"Training dataset length: {len(train)}")
print(f"Validation dataset length: {len(val)}")

train_loader = dataset_loader.get_data_loader(is_binary=True, dataset=train, batch_size=batch_size)
val_loader = dataset_loader.get_data_loader(is_binary=True, dataset=val, batch_size=batch_size)


############################################################
####               BUILD AND TRAIN MODEL                ####
############################################################

model = CNN_model.Binary_CNN(input_dim)
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

except Exception:
    traceback.print_exc()
    # print(e)
