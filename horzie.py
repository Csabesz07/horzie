import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from helper_functions import conversion, standardize, StringHolder
from helper_variables import place_match, sex_match

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split

results = pd.read_csv('./races_2003_2025.csv')

print(results.shape)

results.drop(columns=[
  'date',
  'start_time',
  'versenykiiras',
  'race_name', 
  'versenydij', 
  'program_number', 
  'color',
  'time', 
  'dam'], 
  inplace=True)

# @TODO Try to fill the nullish values (race_time), instead of masking
# print(results.isnull().sum()) # Should check nan as well

results['race_of_the_day'] = results['race_of_the_day'].apply(conversion, args=(place_match,))
results['place'] = results['place'].apply(conversion, args=(place_match,))
results['sex'] = results['sex'].apply(conversion, args=(sex_match,))

mask = results['race_time'].notna()
results['race_time'] = results.loc[mask, 'race_time'] = pd.to_timedelta(
  '00:' + results.loc[mask, 'race_time'].astype(str)
)
results['race_time'] = results['race_time'].dt.total_seconds()

jocky_names = StringHolder(results['jockey'])
horse_names = StringHolder(results['horse_name'])
trainer_names = StringHolder(results['trainer'])
stable_names = StringHolder(results['stable'])
sire_names = StringHolder(results['sire'])

results['jockey'] = results['jockey'].apply(conversion, args=(jocky_names.names,))
results['horse_name'] = results['horse_name'].apply(conversion, args=(horse_names.names,))
results['trainer'] = results['trainer'].apply(conversion, args=(trainer_names.names,))
results['stable'] = results['stable'].apply(conversion, args=(stable_names.names,))
results['sire'] = results['sire'].apply(conversion, args=(sire_names.names,))

results[[
  'race_time', 'distance', 'dividend', 'jockey', 'horse_name', 'trainer', 'stable', 'sire'
  ]] = results[[
    'race_time', 'distance', 'dividend', 'jockey', 'horse_name', 'trainer', 'stable', 'sire'
    ]].apply(standardize)

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

tensor = torch.tensor(results.values)
dataset = TensorDataset(tensor)

data_train, rest = train_test_split(dataset, train_size=0.6)
data_val, data_test = train_test_split(rest, train_size=0.5)

trainloader = DataLoader(data_train, batch_size=2000)
valloader = DataLoader(data_val, batch_size=2000)
testloader = DataLoader(data_test, batch_size=2000)
