import sys
import os
import pandas as pd
import tqdm

file = 'whitelist.txt'

def add_to_whitelist(whitelisted_dataset):
    whitelisted_ids = whitelisted_dataset['id'].unique()

    #print(whitelisted_ids)
    #print(type(whitelisted_ids))

    if(os.path.exists(file) and os.stat(file).st_size != 0) :
        with open(file, 'a') as f:
            old_whitelisted = []
            for line in f: 
                old_whitelisted.append(line)
            whitelisted_ids = whitelisted_ids - old_whitelisted
            for id in whitelisted_ids:
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

    #print(whitelisted)
    rows = dataset.loc[-dataset['id'].isin(whitelisted)]
    #blacklisted.append(rows, ignore_index = True)
    blacklisted = pd.DataFrame(rows)
    #print(rows)
    print(blacklisted)
    print(blacklisted['id'].unique())

    
    

DATA_PATH = 'data/raw.csv'
sys.path.append(DATA_PATH)

colnames = ['time', 'can', 'id', 'dlc', 'payload']

whitelisted_dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)
#add_to_whitelist(whitelisted_dataset)

check_ids(whitelisted_dataset)