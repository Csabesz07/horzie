import torch
import numpy as np
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from custom_model import CustomModel

def nnTraining(dataset, output_size):
  indices = np.arange(len(dataset))
  train_idx, rest_idx = train_test_split(indices, train_size=0.6, shuffle=True)
  val_idx, test_idx   = train_test_split(rest_idx, train_size=0.5, shuffle=True)

  train_dataset = Subset(dataset, train_idx)
  val_dataset   = Subset(dataset, val_idx)
  test_dataset  = Subset(dataset, test_idx)

  dataloader_train = DataLoader(train_dataset, batch_size=2000, shuffle=True)
  dataloader_val = DataLoader(val_dataset, batch_size=2000, shuffle=True)
  dataloader_test = DataLoader(test_dataset, batch_size=2000, shuffle=False)
  
  model = CustomModel(in_features=13, output_size=output_size)
  lossfn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  '''Trainin loop with early stopping'''

  num_epochs = 200
  patience = 20
  best_val_loss = float('inf')
  patience_counter = 0
  best_model_wts = None

  train_losses = []
  val_losses = []

  for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader_train:
      model.train()
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = lossfn(outputs, y_batch)
      loss.backward()
      optimizer.step()

    train_losses.append(loss.mean().item())

    '''Validation part'''
    model.eval()
    with torch.no_grad():
      for x_batch, y_batch in dataloader_val:
        val_outputs = model(x_batch)
        val_loss = lossfn(val_outputs, y_batch)

    val_losses.append(val_loss.mean().item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    best_model_wts = model.state_dict()

  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('Loss Over Epochs')
  plt.savefig('./docs/loss.png', bbox_inches='tight')  

  best_model = model.load_state_dict(best_model_wts)
