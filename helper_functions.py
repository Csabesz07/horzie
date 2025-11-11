__place_match = {
  'I.':1,
  'II.':2,
  'III.':3,
  'IV.':4,
  'V.':5,
  'VI.':6,
  'VII.':7,
  'VIII.':8,
  'IX.':9,
  'X.':10,
  'XI.':11,
  'XII.':12,
  'disq.':0,
  '0.':0,
  '1.':1,
  '2.':2,
  '3.':3,
  '4.':4,
  '5.':5,
  '6.':6,
  '7.':7,
  '8.':8,
  '9.':9,
  '10.':10,
  '11.':11,
  '12.':12,
}

def place_conversion(x):
  for [key, val] in __place_match.items():
    if key in str(x):
      x = val
      return x
  return str(x) 

def standardize(x):
  return (x - x.min()) / (x.max() - x.min())