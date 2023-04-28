import sys
import os
import pandas as pd
import numpy as np
import tqdm

DEBUG = False

FILENAME = 'whitelist.txt'
CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file = (os.path.join(CUR_PATH, FILENAME))

def add_to_whitelist(whitelisted_dataset):
    whitelisted_ids = whitelisted_dataset['id'].unique()
    if(os.path.exists(file) and os.stat(file).st_size != 0):
        # ugly but it works
        with open(file, 'r+') as f:
            old_whitelisted = []
            for line in f: old_whitelisted.append(line.strip())
            new_whitelisted = list(set(whitelisted_ids) - set(old_whitelisted))
            for id in new_whitelisted:
                print('Added ID:', id)
                f.write(id + '\n')
    else:
        with open(file, 'w') as f:
            for id in whitelisted_ids:
                f.write(id + '\n')

def get_blacklist(dataset):
    if(os.path.exists(file) and os.stat(file).st_size != 0) :
        with open(file, 'r') as f:
            whitelisted = []
            for line in f: 
                whitelisted.append(line.strip())

    blacklisted_dataset = dataset.loc[-dataset['id'].isin(whitelisted)]
    whitelisted_dataset = dataset.loc[dataset['id'].isin(whitelisted)]
    # debug prints
    if DEBUG:
        print(dataset.size, whitelisted_dataset.size, blacklisted_dataset.size)
        print(len(whitelisted_dataset['id'].unique()), len(blacklisted_dataset['id'].unique()))
    return whitelisted_dataset, blacklisted_dataset

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../../data/raw.csv'
    DATA_PATH = (mod_path / relative_path).resolve()  # 'data/raw.csv'
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)

    # add_to_whitelist(dataset)
    get_blacklist(dataset)