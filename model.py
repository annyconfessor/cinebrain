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
      nn.Linear(16,1)
    )

  def forward(self, x):
    return self.fc(x)
  
model = RecommendedMovies(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
  optimizer.zero_grad()
  outputs = model(X)
  loss = criterion(outputs, X)
  loss.backward()
  optimizer.step()
  if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():4f}")
  