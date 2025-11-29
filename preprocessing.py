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

  preserved_horse_names = Path('./docs/horse_names.csv')
  if preserved_horse_names.exists():
    update_horse_names(horse_names)
  else:
    with open('./docs/horse_names.csv', mode='w', newline='') as file:
      w = csv.DictWriter(file, ["horse_name","id"])
      w.writeheader()
      w.writerows([{name, idx} for idx, name in enumerate(horse_names.names)])

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

def update_horse_names(data):
  with open('./docs/horse_names.csv', mode='w', newline='') as file:
      w = csv.DictWriter(file, ["horse_name","id"])
      w.writeheader()
      w.writerows(data.names)

def vectorize_data(data):
  '''
  Load the data into a tensor format of the following:
  [
    [feat1_race1, feat2_race1, ..., featK_race1],
    [feat1_race2, feat2_race2, ..., featK_race2],
    ...
    [feat1_raceN, feat2_raceN, ..., featK_raceN]
  ]
  '''

