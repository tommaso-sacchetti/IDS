import sys
import os
import pandas as pd
import numpy as np
import tqdm

file = 'whitelist.txt'

def add_to_whitelist(whitelisted_dataset):
    whitelisted_ids = whitelisted_dataset['id'].unique()

    # ugly way to insert unique IDs but as prototype it works
    # only alternative that's in my mind would be to cycle over all values 
    # and for every value look if it's present in the whitelist file 
    # more computational expensive but way less memory expensive
    if(os.path.exists(file) and os.stat(file).st_size != 0) :
        with open(file, 'r+') as f:
            old_whitelisted = []
            for line in f: 
                old_whitelisted.append(line.strip())
            whitelisted_ids = list(set(whitelisted_ids) - set(old_whitelisted))
            for id in whitelisted_ids:
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
    #print(dataset.size, whitelisted_dataset.size, blacklisted_dataset.size)
    #print(len(whitelisted_dataset['id'].unique()), len(blacklisted_dataset['id'].unique()))
    return whitelisted_dataset, blacklisted_dataset

    
DATA_PATH = 'data/raw.csv'

colnames = ['time', 'can', 'id', 'dlc', 'payload']

whitelisted_dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)
print(len(whitelisted_dataset['id'].unique()))
#add_to_whitelist(whitelisted_dataset)

get_blacklist(whitelisted_dataset)