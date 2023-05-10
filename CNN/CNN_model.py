import math
import torch.nn as nn
# from torchsummary import summary

def compute_output_shape(input, kernel, padding=0, stride=1, convolutional=True):
  if convolutional:
    return math.floor(((input - kernel + 2*padding)/stride)+1) 
  else:
    return math.ceil(((input - kernel + 2*padding)/stride)+1)

class Binary_CNN(nn.Module):
  def __init__(self, input_dim=10, in_channels=1, output_dim=1, kernel = 2, conv_channels = 512):
    super().__init__()
    output_conv1 = compute_output_shape(input_dim, kernel)
    output_maxpool = compute_output_shape(output_conv1, kernel, stride=kernel)
    input_fc1 = output_maxpool * conv_channels
    # for convolutional output size: floor((h+2*p-k)/s)+1
    # for pooling output size: ceil((h+2*p-k)/s)+1
    self.conv1 = nn.Conv1d(in_channels, out_channels=conv_channels, kernel_size=kernel)
    self.tanh = nn.Tanh()
    self.dropout = nn.Dropout(0.1)
    self.maxpool = nn.MaxPool1d(kernel_size=kernel)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(input_fc1, output_dim)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.conv1(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.sigmoid(x)
    return x

class Multiclass_CNN(nn.Module):
  def __init__(self, input_dim=10, in_channels=1, output_dim=5, kernel = 2, conv_channels = 512):
    super().__init__()
    output_conv1 = compute_output_shape(input_dim, kernel)
    #output_conv2 = compute_output_shape(output_conv1, kernel)
    output_maxpool = compute_output_shape(output_conv1, kernel, stride=kernel)
    input_fc1 = output_maxpool * conv_channels
    print(input_fc1)
    
    #conv_channels = 512
    self.conv1 = nn.Conv1d(in_channels, out_channels=conv_channels, kernel_size=kernel)
    self.sigmoid = nn.Sigmoid()
    self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel)
    self.dropout = nn.Dropout(0.1)
    self.maxpool = nn.MaxPool1d(kernel_size=kernel)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(input_fc1, output_dim)
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = self.conv1(x)
    x = self.sigmoid(x)
    x = self.conv2(x)
    x = self.sigmoid(x)
    x = self.dropout(x)
    x = self.maxpool(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.sigmoid(x)
    return x

# model = binary_CNN()
# summary(model, (1, 10))
# model = multiclass_CNN()
# summary(model, (1, 10))