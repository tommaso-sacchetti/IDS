import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

DEBUG = False

############################################################
####               ID BLACKLIST FILTERING               ####
############################################################

whitelist_filename = 'rules/whitelist.txt'
whitelist_file = (os.path.join(CUR_PATH, whitelist_filename))

def add_to_whitelist(whitelisted_dataset):
    whitelisted_ids = whitelisted_dataset['id'].unique()
    if(os.path.exists(whitelist_file) and os.stat(whitelist_file).st_size != 0):
        # ugly but it works
        with open(whitelist_file, 'r+') as f:
            old_whitelisted = []
            for line in f: old_whitelisted.append(line.strip())
            new_whitelisted = list(set(whitelisted_ids) - set(old_whitelisted))
            for id in (pbar := tqdm(new_whitelisted)):
                pbar.set_description(f'Adding ID to whitelist: {id}')
                f.write(id + '\n')
    else:
        with open(whitelist_file, 'w') as f:
            for id in (pbar := tqdm(whitelisted_ids)):
                pbar.set_description(f'whitelisting IDs')
                f.write(id + '\n')

def filter_blacklisted_id(dataset):
    if(os.path.exists(whitelist_file) and os.stat(whitelist_file).st_size != 0) :
        with open(whitelist_file, 'r') as f:
            whitelisted = []
            for line in (pbar := tqdm(f)):
                pbar.set_description("filtering blacklisted IDs")
                whitelisted.append(line.strip())
        blacklisted_dataset = dataset.loc[-dataset['id'].isin(whitelisted)]
        whitelisted_dataset = dataset.loc[dataset['id'].isin(whitelisted)]
        # debug prints
        if DEBUG:
            print(f'Dataset shape: {dataset.shape}; whitelisted_dataset shape: {whitelisted_dataset.shape}; blacklisted_dataset shape: {blacklisted_dataset.size}')
            print(f"IDs in whitelist: {len(whitelisted_dataset['id'].unique())}; IDs in blacklist: {len(blacklisted_dataset['id'].unique())}")
            print('Blacklisted IDs:', blacklisted_dataset['id'].unique())
        return whitelisted_dataset, blacklisted_dataset
    else:
        print("Error: no whitelist available. Returning full dataset")
        return dataset, pd.DataFrame()


############################################################
####                    TIME INTERVAL                   ####
############################################################

THRESHOLD = 20
K = 5
periods_filename = 'rules/periods.npy'
periods_file = (os.path.join(CUR_PATH, periods_filename))

# TODO: STORE ONLY RANGE_MIN AND RANGE_MAX, dict might be good (not the periods, waste of memory)
def store_periods(dataset):
    id_periods = dict()
    for id in (pbar := tqdm(dataset['id'].unique())):
        pbar.set_description("saving IDs periods")
        id_packets = dataset.loc[dataset['id'] == id]
        times_of_arrival = id_packets['time'].to_numpy()
        periods = np.diff(times_of_arrival)
        # inserd ID only if periodic
        if(periods.size != 0):
            if check_periodicity(periods, THRESHOLD):
                id_periods[id] = periods
    # Save periods of periodic IDs in file            
    np.save(periods_file, id_periods)
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

def check_dataset_time_intervals(dataset: pd.DataFrame):
    id_periods = dict()
    id_periods = np.load(periods_file, allow_pickle='TRUE').item()
    blacklisted_dataset = pd.DataFrame()
    whitelisted_dataset = pd.DataFrame()
    for id in (pbar := tqdm(dataset['id'].unique())):
        pbar.set_description("Checking packet intervals")
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
    if DEBUG:
        print('\n DATASET: ',blacklisted_dataset, '\n\n')
        print(whitelisted_dataset.size, blacklisted_dataset.size)
    whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return whitelisted_dataset, blacklisted_dataset

# debug purpose
def plot_periods(id: str, dataset: pd.DataFrame):
    id_packets = dataset.loc[dataset['id'] == id]
    times_of_arrival = id_packets['time'].to_numpy()
    periods = np.diff(times_of_arrival) 
    plt.hist(periods, color='lightgreen', ec='black', bins=100)
    r_min, r_max = compute_range(K, periods)
    plt.axvline(r_min, 0, 1, label='min')
    plt.axvline(r_max, 0, 1, label='max')
    plt.show()


############################################################
####                      DLC CHECK                     ####
############################################################

def check_dlc(dataset):
    whitelisted_dataset = pd.DataFrame()
    blacklisted_dataset = pd.DataFrame()
    # using numpy array for faster computing time
    dlc = dataset['dlc'].to_numpy()
    payload = dataset['payload'].to_numpy()
    print("Checking data length code...")
    def func(x):
        return len(x)/8
    payload = np.vectorize(func)(payload)
    diff = np.subtract(dlc, payload)
    indexes = np.argwhere(diff).flatten()
    blacklisted_dataset = dataset.iloc[indexes]
    if len(blacklisted_dataset) != 0:
        whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    else: whitelisted_dataset = dataset
    return whitelisted_dataset, blacklisted_dataset


############################################################
####                RULE-BASED FILTERING                ####
############################################################

def initialize_rules(dataset):
    add_to_whitelist(dataset)
    store_periods(dataset)

def filter(dataset):
    id_whitelist, id_blacklist = filter_blacklisted_id(dataset)
    period_whitelist, period_blacklist = check_dataset_time_intervals(id_whitelist)
    dlc_whitelist, dlc_blacklist = check_dlc(period_whitelist)

    return dlc_whitelist, id_blacklist, period_blacklist, dlc_blacklist


############################################################
####                      TESTING                       ####
############################################################

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../data/raw.csv'
    data_path = (mod_path / relative_path).resolve()
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(data_path, names=colnames, header=None)
    initialize_rules(dataset)
    #a, b, c, d = filter(dataset)
    #print(len(a), len(b), len(c), len(d))
    #print(len(a) + len(b) + len(c) + len(d), len(dataset))