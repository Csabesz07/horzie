import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.display import Image as IMG

'''
The helper functions
'''

place_match = {
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
  for [key, val] in place_match.items():
    if key in str(x):
      x = val
      return x
  return str(x)

'''
The magic itself
'''

results = pd.read_csv('./races_2003_2025.csv')

# print(results.shape) # Uncomment to see the shape of the result set

results.drop(columns=[
  'date',
  'start_time',
  'versenykiiras',
  'race_name', 
  'versenydij', 
  'program_number', 
  'horse_name', 
  'color', 
  'sire', 
  'dam'], 
  inplace=True)

# @TODO Try to fill the nullish values

results = results.map(place_conversion)
results['place'] = pd.to_numeric(results['place'], errors='coerce')
results = results.map(lambda v: float(v) if str(v).isnumeric() else str(v))

print(results.info()) # Uncomment to see information about the results
pd.set_option('display.max_colwidth', None)
print(results.head()) # Uncomment to see first 5 elements of the result set

# sns.set_theme(rc = {'figure.figsize':(20,20), 'font.weight': 'bold', 'font.size': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'xtick.top': True, 'xtick.labeltop': True})
# sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
# ax = sns.heatmap(results.corr().round(2), annot=True, cmap="coolwarm")
# plt.savefig('corr.png', bbox_inches='tight')