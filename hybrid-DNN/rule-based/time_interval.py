import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt


# TODO: potentially add an online method for computing standard deviation

THRESHOLD = 20
K = 5
output_filename = 'periods.npy'

def store_periods(dataset):
    cur_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    output_file = (os.path.join(cur_path, output_filename))
    id_periods = dict()
    
    for id in dataset['id'].unique():
        # print(id)
        id_packets = dataset.loc[dataset['id'] == id]
        times_of_arrival = id_packets['time'].to_numpy()
        periods = np.diff(times_of_arrival)
        # insert id in dict only if periodic
        print(id)
        print(periods)
        if(periods.size != 0):
            if check_periodicity(periods, THRESHOLD):
                id_periods[id] = periods

    np.save(output_file, id_periods)
    return id_periods

def check_periodicity(periods: np.array, percentual_threshold: int):
    avg = periods.mean()
    sigma = periods.std()
    coefficient = sigma / avg
    return sigma < percentual_threshold / 100 

def compute_range(k: int, periods: np.array):
    avg = periods.mean()
    sigma = periods.std()
    range_min = avg - 5*sigma
    range_max = avg + 5*sigma
    return range_min,range_max

def check_dataset_time_intervals(dataset):
    id_periods = dict()
    cur_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    output_file = (os.path.join(cur_path, output_filename))
    id_periods = np.load(output_file, allow_pickle='TRUE').item()

    for id in dataset['id'].unique():
        if id not in id_periods:
            print(id, ' is not periodic')
            print('ITERATION')
            # id is not periodic, no way to check it's correct periodicity
            continue
        else:
            print('ITERATION')
            # compute the periods
            id_packets = dataset.loc[dataset['id'] == id]
            times_of_arrival = id_packets['time'].to_numpy()
            periods = np.diff(times_of_arrival)
            range_min, range_max = compute_range(K, id_periods[id])  

            df = dataset.loc[dataset['id'] == id]
            df = df.rename_axis(None)
            blacklisted_dataset = pd.DataFrame()
            whitelisted_dataset = pd.DataFrame()
            for index, period in enumerate(periods):
                if period > range_max or period < range_min:
                    # since first row with id of df is not considered in the deltas 
                    # (no way to compute it) it's ignored and the +1 is added
                    row = df.iloc[[index+1]]
                    print(row)
                    blacklisted_dataset = pd.concat([blacklisted_dataset, row])
                    
    whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    print(whitelisted_dataset.size, blacklisted_dataset.size)
    return whitelisted_dataset, blacklisted_dataset

def plot():
    id_periods = dict()
    cur_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    output_file = (os.path.join(cur_path, output_filename))
    id_periods = np.load(output_file, allow_pickle='TRUE').item()

    print(id_periods.keys())

    data = id_periods['1F2']
    plt.hist(data, color='lightgreen', ec='black', bins=100)
    plt.show()


mod_path = Path(__file__).parent
relative_path = '../../data/raw.csv'
DATA_PATH = (mod_path / relative_path).resolve()  # 'data/raw.csv'

colnames = ['time', 'can', 'id', 'dlc', 'payload']
dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)

# dictionary = store_periods(dataset)

a, b = check_dataset_time_intervals(dataset)
print(b)
#plot()


