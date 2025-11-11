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

def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

'''
The magic itself
'''

results = pd.read_csv('./races_2003_2025.csv')

print(results.shape)

results.drop(columns=[
  'date',
  'start_time',
  'versenykiiras',
  'race_name', 
  'versenydij', 
  'program_number', 
  'horse_name', 
  'color',
  'time', 
  'sire', 
  'dam'], 
  inplace=True)

# @TODO Try to fill the nullish values (race_time), instead of masking
# print(results.isnull().sum()) # Should check nan as well

results['race_of_the_day'] = results['race_of_the_day'].apply(place_conversion)


results['place'] = results['place'].apply(place_conversion)
results['place'] = pd.to_numeric(results['place'], errors='coerce')

mask = results['race_time'].notna()
results['race_time'] = results.loc[mask, 'race_time'] = pd.to_timedelta(
    '00:' + results.loc[mask, 'race_time'].astype(str)
)
results['race_time'] = results['race_time'].dt.total_seconds()
results[['race_time', 'distance', 'dividend']] = results[['race_time', 'distance', 'dividend']].apply(normalize)

# @TODO Try to convert string columns like trainer to a numeric value, as it might mean mething who trained the horses

print(results.info())
pd.set_option('display.max_colwidth', None)
print(results.head())

X = results
Y = results['place']

sns.set_theme(rc = {'figure.figsize':(20,20), 'font.weight': 'bold', 'font.size': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'xtick.top': True, 'xtick.labeltop': True})
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
ax = sns.heatmap(results.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm")
plt.savefig('corr.png', bbox_inches='tight')