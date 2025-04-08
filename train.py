import torch
from model import optimizer, model, criterion, X

for epoch in range(100):
  optimizer.zero_grad()
  outputs = model(X)
  loss = criterion(outputs, X)
  loss.backward()
  optimizer.step()
  if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():4f}")

torch.save(model.state_dict(), "model.pth")