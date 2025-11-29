import pandas as pd
from helper_functions import conversion, NameHolder, standardize
from helper_variables import place_match, sex_match
import csv
from pathlib import Path

def preprocess_data(data) -> pd.DataFrame:
  data.drop(columns=[
  'date',
  'race_time',
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

  data['race_of_the_day'] = data['race_of_the_day'].apply(conversion, args=(place_match,))
  data['place'] = data['place'].apply(conversion, args=(place_match,))
  data['place'] = data.loc[data['place'] > 0, ['place']]
  data = data[data['place'].notna()]
  data['sex'] = data['sex'].apply(conversion, args=(sex_match,))

  jocky_names = NameHolder(data['jockey'])
  horse_names = NameHolder(data['horse_name'])

  if Path('./docs/horse_names.csv').exists():
    old_df = pd.read_csv('./docs/horse_names.csv')
    old_names = dict(zip(old_df['horse_name'], old_df['id']))
    missing_names = [name for name in horse_names.names if name not in old_names]

    if len(missing_names) > 0:
      max_id = max(old_names.values()) if old_names else 0
      update_horse_names(missing_names, max_id)
  else:
    with open('./docs/horse_names.csv', mode='w', newline='', encoding='utf-8') as file:
      w = csv.DictWriter(file, ["horse_name","id"])
      w.writeheader()
      w.writerows([{"horse_name": name, "id": idx} for idx, name in enumerate(horse_names.names.keys(), start=1)])

  trainer_names = NameHolder(data['trainer'])
  stable_names = NameHolder(data['stable'])
  sire_names = NameHolder(data['sire'])

  data['jockey'] = data['jockey'].apply(conversion, args=(jocky_names.names,))
  data['horse_name'] = data['horse_name'].apply(conversion, args=(horse_names.names,))
  data['trainer'] = data['trainer'].apply(conversion, args=(trainer_names.names,))
  data['stable'] = data['stable'].apply(conversion, args=(stable_names.names,))
  data['sire'] = data['sire'].apply(conversion, args=(sire_names.names,))

  data[[
    'distance', 'dividend', 'jockey', 'horse_name', 'trainer', 'stable', 'sire'
    ]] = data[[
      'distance', 'dividend', 'jockey', 'horse_name', 'trainer', 'stable', 'sire'
      ]].apply(standardize)
  
  return data

def update_horse_names(data, start_id):
  curr_id = start_id + 1
  dict_names = {}
  for _, name in enumerate(data):
    dict_names[name] = curr_id
    curr_id += 1

  with open('./docs/horse_names.csv', mode='a', newline='', encoding='utf-8') as file:
    w = csv.DictWriter(file, ["horse_name","id"])
    w.writerows([{"horse_name": name, "id": dict_names[name]} for _, name in enumerate(dict_names.keys())])

def vectorize_data(dataframe):
  '''
    The desired format of the dataset is a list of horses,
    with the list of their races and feautres
  '''

  print('now vectorizing')
  raise NotImplementedError('Vectorization not implemented yet')

  horses = {}
  for _, row in dataframe.iterrows():
    horse_name = row['horse_name']
    if horse_name in horses:
      horses[horse_name].append(row)
    else:
      horses[horse_name] = [row.to_numpy()]