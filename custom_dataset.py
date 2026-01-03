import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  input_size = 0

  def __init__(self, dataset):
    self.dataset = dataset
    CustomDataset.input_size = len(dataset[0][0]) - 1

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    curr_data = self.dataset[index]
    input_sequence = torch.tensor([race[:-1] for race in curr_data], dtype=torch.float32)
    place = int(curr_data[-1][-1])
    target_class = place if 1 <= place <= 5 else 0
    target_val = torch.tensor(target_class, dtype=torch.long)
    return input_sequence, target_val
    