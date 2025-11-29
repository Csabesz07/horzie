import pandas as pd

def conversion(x, collection):
  for [key, val] in collection.items():
    if key in str(x):
      x = val
      return x
  return str(x) 

def standardize(x):
  return (x - x.min()) / (x.max() - x.min())

def parse_race_time(t):
  if pd.isna(t) or not isinstance(t, str):
    return float('nan')

  try:
      minutes, seconds = t.split(':')
      return int(minutes) * 60 + float(seconds)
  except Exception:
      return float('nan')

class NameHolder:
  def __init__(self, collection):
    self.names = {}

    for key in collection:
      if key not in self.names:
        self.names[key] = len(self.names.keys()) + 1
        
class EarlyStopper:
  def __init__(self, patience=5, min_delta=0):
    self.early_stopped = False
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = float('inf')

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stopped = True
        return True
      
    return False