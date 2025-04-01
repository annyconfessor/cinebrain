import torch.nn as nn
import torch.optim as optim
from model import SimpleNN

model = SimpleNN(input_size=20, num_classes=2)

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


