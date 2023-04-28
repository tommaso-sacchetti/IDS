import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import time


# TODO: potentially add an online method for computing standard deviation

THRESHOLD = 20
K = 5
OUTPUT_FILENAME = 'periods.npy'
CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
output_file = (os.path.join(CUR_PATH, OUTPUT_FILENAME))

DEBUG = True

# TODO: STORE ONLY RANGE_MIN AND RANGE_MAX (not the periods, waste of memory)
def store_periods(dataset):
    id_periods = dict()
    for id in tqdm(dataset['id'].unique()):
        id_packets = dataset.loc[dataset['id'] == id]
        times_of_arrival = id_packets['time'].to_numpy()
        periods = np.diff(times_of_arrival)
        # inserd ID only if periodic
        if(periods.size != 0):
            if check_periodicity(periods, THRESHOLD):
                id_periods[id] = periods
    # Save periods of periodic IDs in file            
    np.save(output_file, id_periods)
    return id_periods

def check_periodicity(periods: np.array, percentual_threshold: int):
    avg = periods.mean()
    sigma = periods.std()
    coefficient = sigma / avg
    percentual_threshold = percentual_threshold / 100
    return coefficient < percentual_threshold

def compute_range(k: int, periods: np.array):
    avg = periods.mean()
    sigma = periods.std()
    range_min = avg - k*sigma
    range_max = avg + k*sigma
    return range_min,range_max

def check_dataset_time_intervals(dataset):
    id_periods = dict()
    id_periods = np.load(output_file, allow_pickle='TRUE').item()
    blacklisted_dataset = pd.DataFrame()
    whitelisted_dataset = pd.DataFrame()

    for id in tqdm(dataset['id'].unique()):
        if id in id_periods:
            # compute the periods
            id_packets = dataset.loc[dataset['id'] == id]
            times_of_arrival = id_packets['time'].to_numpy()
            periods = np.diff(times_of_arrival)
            range_min, range_max = compute_range(K, id_periods[id])  
            df = dataset.loc[dataset['id'] == id]
            for index, period in enumerate(periods):
                if period > range_max or period < range_min:
                    # since first row with id of df is not considered in the deltas 
                    # (no way to compute it) it's ignored and the +1 is added
                    if DEBUG: print(f'ID: {id}, MIN: {range_min}, MAX: {range_max}, PERIOD {period}')
                    row = df.iloc[[index]]
                    blacklisted_dataset = pd.concat([blacklisted_dataset, row])
        else:
            # id is not periodic, no way to check it's correct periodicity
            if DEBUG: print(id, ' is not periodic')
            continue

    print('\n DATASET: ',blacklisted_dataset, '\n\n')
    whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    print(whitelisted_dataset.size, blacklisted_dataset.size)
    return whitelisted_dataset, blacklisted_dataset

def plot_periods(id: string, dataset: DataFrame):
    id_packets = dataset.loc[dataset['id'] == id]
    times_of_arrival = id_packets['time'].to_numpy()
    periods = np.diff(times_of_arrival)
    
    plt.hist(periods, color='lightgreen', ec='black', bins=100)
    r_min, r_max = compute_range(K, periods)
    plt.axvline(r_min, 0, 1, label='min')
    plt.axvline(r_max, 0, 1, label='max')
    plt.show()

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../../data/raw.csv'
    DATA_PATH = (mod_path / relative_path).resolve()  # 'data/raw.csv'
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)

    a, b = check_dataset_time_intervals(dataset)
    print(a.shape, b.shape)



