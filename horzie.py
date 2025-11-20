import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from helper_functions import conversion, standardize, NameHolder
from helper_variables import place_match, sex_match
from nn_training import nnTraining

results = pd.read_csv('./docs/races_2003_2025.csv')

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
results['place'] = results.loc[results['place'] > 0, ['place']]
results = results[results['place'].notna()]
results['sex'] = results['sex'].apply(conversion, args=(sex_match,))

mask = results['race_time'].notna()
results['race_time'] = results.loc[mask, 'race_time'] = pd.to_timedelta(
  '00:' + results.loc[mask, 'race_time'].astype(str)
)
results['race_time'] = results['race_time'].dt.total_seconds()

jocky_names = NameHolder(results['jockey'])
horse_names = NameHolder(results['horse_name'])
trainer_names = NameHolder(results['trainer'])
stable_names = NameHolder(results['stable'])
sire_names = NameHolder(results['sire'])

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
plt.savefig('./docs/corr.png', bbox_inches='tight')

'''
I will use 3 datasets:
  - Train
  - Validation
  - Test
In a ratio of 6-2-2
'''

X = results.drop(columns=['place']).to_numpy()
Y = results['place'].to_numpy()

print(results.shape, X.shape, Y.shape)

x_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(Y, dtype=torch.long)
dataset = TensorDataset(x_tensor, y_tensor)

nnTraining(dataset=dataset, output_size=len(set(place_match.values())))