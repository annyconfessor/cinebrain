import torch
import torch.nn as nn
import torch.optim as optim
from config import X

class RecommendedMovies(nn.Module):
  def __init__(self, input_size):
    super(RecommendedMovies, self).__init__()
    self.fc = nn.Sequential(
      nn.Linear(input_size, 16),
      nn.ReLU(),
      nn.Linear(16,1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.fc(x) * 10
  
model = RecommendedMovies(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)