import torch
import numpy as np
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

from custom_model import CustomModel
from custom_dataset import CustomDataset
from helper_functions import EarlyStopper

def nnTraining(dataset, output_size):  
  '''
    I will use 3 datasets:
      - Train
      - Validation
      - Test
    In a ratio of 6-2-2
  '''
  train, rest = train_test_split(dataset, train_size=0.6, shuffle=True)
  val, test = train_test_split(rest, train_size=0.5, shuffle=True)

  train_dataset = CustomDataset(train)
  val_dataset = CustomDataset(val)
  test_dataset  = CustomDataset(test)

  # The bacth size refers to the number of horses in a batch
  dataloader_train = DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
  dataloader_val = DataLoader(val_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
  dataloader_test = DataLoader(test_dataset, batch_size=200, shuffle=False, collate_fn=collate_fn)
  
  input_size = CustomDataset.input_size
  hidden_size = 64
  num_layers = 1
  output_size = 1
  learning_rate = 0.001
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = CustomModel(input_size, hidden_size, num_layers, output_size, device).to(device)
  lossfn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  '''Trainin loop with early stopping'''
  num_epochs = 200
  patience = 20
  best_model_wts = None

  train_losses = []
  val_losses = []

  early_stopper = EarlyStopper(patience=patience, min_delta=0.001)

  for epoch in range(num_epochs):
    '''Training part'''
    loss = do_training(model, lossfn, optimizer, dataloader_train)
    train_losses.append(loss.mean().item())

    '''Validation part'''
    val_loss = do_validation(model, lossfn, dataloader_val)
    val_losses.append(val_loss.mean().item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    best_model_wts = model.state_dict()

    if early_stopper.early_stop(val_loss.item()):
      print(f"Early stopping at epoch {epoch}")
      break

  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_losses, label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.title('Loss Over Epochs (Early stopped)' if early_stopper.early_stopped else 'Loss Over Epochs')
  plt.savefig('./docs/loss_latest.png', bbox_inches='tight')  

  best_model = model.load_state_dict(best_model_wts)

def do_training(model, lossfn, optimizer, dataloader):
  for x_batch, lengths, y_batch in dataloader:
    model.train()
    optimizer.zero_grad()
    outputs = model(x_batch, lengths)
    loss = lossfn(outputs, y_batch)
    loss.backward()
    optimizer.step()

  return loss

def do_validation(model, lossfn, dataLoader):
  model.eval()
  with torch.no_grad():
    for x_batch, lengths, y_batch in dataLoader:
      val_outputs = model(x_batch, lengths)
      val_loss = lossfn(val_outputs, y_batch)

  return val_loss

def collate_fn(batch):
  sequences, targets = zip(*batch)
  seq_lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
  padded_sequences = pad_sequence(sequences, batch_first=True)
  targets = torch.stack(targets)

  return padded_sequences, seq_lengths, targets