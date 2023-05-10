import torch.nn as nn

class DNN(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(DNN, self).__init__()
    # 5 input features, binary output
    self.fc1 = nn.Linear(input_dim, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 80)
    self.fc4 = nn.Linear(80, 60)
    self.fc5 = nn.Linear(60, 40)
    self.out = nn.Linear(40, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.sigmoid(self.out(x))
    return x

net = DNN(input_dim, output_dim)
print(net)