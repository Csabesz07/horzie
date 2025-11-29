import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from helper_variables import place_match
from preprocessing import preprocess_data, vectorize_data
from nn_training import nnTraining

results = pd.read_csv('./docs/races_2003_2025.csv')

print(results.shape)

data = preprocess_data(results)

print(data.info())
pd.set_option('display.max_colwidth', None)
print(data.head())

sns.set_theme(rc = {'figure.figsize':(20,20), 'font.weight': 'bold', 'font.size': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'xtick.top': True, 'xtick.labeltop': True})
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
ax = sns.heatmap(data.corr(numeric_only=True).round(2), annot=True, cmap="coolwarm")
plt.savefig('./docs/corr.png', bbox_inches='tight')

dataset = vectorize_data(data)

nnTraining(dataset=dataset, output_size=len(set(place_match.values())))