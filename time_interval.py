import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def calculate_periodicity(dataset):
    id_periods = dict()

    for id in tqdm(dataset['id'].unique()):
        print(id)
        id_packets = dataset.loc[dataset['id'] == id]
        times_of_arrival = id_packets['time'].to_numpy()
        periods = np.diff(times_of_arrival)
        id_periods[id] = periods

    #with open('output.pkl', 'w') as fp:
    #   pickle.dump(id_periods, fp)
    #  print('dictionary saved successfully to file')

    return id_periods


DATA_PATH = 'data/raw.csv'

colnames = ['time', 'can', 'id', 'dlc', 'payload']
dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)

#print(dataset.head())

#print(dataset.loc[dataset['id'] == '1F2'])

dictionary = calculate_periodicity(dataset)
print(dictionary.keys())


with open('output.txt', 'w') as fp:
    for item in dictionary['417']:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')


#TODO: controllare che il numero dei delta corrisponda con il numero di messaggi dell'ID

