import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class CustomModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, device):
    super(CustomModel, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.last_ht = None
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, lengths):
    '''Initial hidden and cell states'''
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

    '''Forward propagate LSTM without the PAD steps'''
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (hn, _) = self.lstm(packed, (h0, c0))
    self.last_ht = hn[-1]

    out = self.fc(self.last_ht)
    return out
