def conversion(x, collection):
  for [key, val] in collection.items():
    if key in str(x):
      x = val
      return x
  return str(x) 

def standardize(x):
  return (x - x.min()) / (x.max() - x.min())

class StringHolder:
  names = {}

  def __init__(self, collection):
    for key in collection:
      if key not in self.names:
        self.names[key] = len(self.names.keys()) + 1
      
