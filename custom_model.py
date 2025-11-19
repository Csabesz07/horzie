from torch import nn

class CustomModel(nn.Module):
  def __init__(self, in_features):
    super(CustomModel, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(in_features, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Linear(16, 1)
    )

  def forward(self, X):
    return self.model(X)