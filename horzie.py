import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from helper_functions import place_conversion, standardize

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader, random_split

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
results[['race_time', 'distance', 'dividend']] = results[['race_time', 'distance', 'dividend']].apply(standardize)

# @TODO Try to convert string columns like trainer to a numeric value, as it might mean mething who trained the horses

print(results.info())
pd.set_option('display.max_colwidth', None)
print(results.head())

sns.set_theme(rc = {'figure.figsize':(20,20), 'font.weight': 'bold', 'font.size': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'xtick.top': True, 'xtick.labeltop': True})
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
ax = sns.heatmap(results.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm")
plt.savefig('corr.png', bbox_inches='tight')

'''
I will use 3 datasets:
  - Train
  - Validation
  - Test
In a ratio of 6-2-2
'''

X = results.loc[ : , results.columns != 'place']
Y = results['place']

transforms = transforms.Compose([transforms.ToTensor()])

