import sys
import os
import time
import pickle
import math
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
hamming_filename = 'pre-process/hamming.npy'
hamming_file = (os.path.join(CUR_PATH, hamming_filename))

def _get_ids(dataset):
    return dataset['id'].to_numpy()

# HAMMING DISTANCE WORKS ONLY IN DATAFIELDS OF SAME LENGTH
def _get_hamming_distances(dataset):
    ids = _get_ids(dataset)
    payloads = dataset['payload'].to_numpy()
    id_payload = list(zip(ids, payloads))
    last_message = dict()
    hamming_distances = np.empty(0, int)
    # id_payload = id_payload[:100]
    start_time = time.time()
    for id, payload in (pbar := tqdm(id_payload)):
        pbar.set_description('Computing Hamming distances')
        if id not in last_message:
            hamming_distances = np.append(hamming_distances, 0)
            last_message[id] = payload
        else:
            distance = hamming(payload, last_message[id])
            hamming_distances = np.append(hamming_distances, distance)
            last_message[id] = payload
    execution_time = time.time() - start_time
    np.save(hamming_file, hamming_distances)
    return hamming_distances

def hamming(s1, s2):
    assert len(s1) == len(s2)
    return bin(int(s1,2)^int(s2,2)).count('1')

def entropy(dataset): 
    distribution = np.zeros(2**8)
    # calculation of the distribution
    payloads = dataset['payload'].to_numpy()
    for payload in (pbar := tqdm(payloads)):
        pbar.set_description('calculating distribution')
        single_bytes = split_string(payload)
        for byte in single_bytes:
            distribution[int(byte,2)] += 1
    distribution = distribution / np.sum(distribution)
    # calculation of entropy
    entropy = np.empty(0, int)
    for payload in (pbar := tqdm(payloads)):
        pbar.set_description('calculating entropy')
        single_bytes = split_string(payload)
        sum = 0
        for byte in single_bytes:
            integer = int(byte,2)
            pk = distribution[integer]
            sum += pk * math.log(pk)
        entropy = np.append(entropy, sum)
    
    plt.hist(entropy, color='lightgreen', ec='black', bins=500)
    plt.show()
    return entropy

def split_string(s):
    chunks, chunk_size = len(s), 8
    return [s[i:i+chunk_size] for i in range(0, chunks, chunk_size)]

# BYTES OF IMPORTANCE: if payloads have different lengths, which bytes should i consider? e.g. i try with first and eight byte, which byte i consider as eight in the payloads that have 5 bytes?

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../data/raw.csv'
    data_path = (mod_path / relative_path).resolve()
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(data_path, names=colnames, header=None)
    index = dataset.index[-1]
    dataset = dataset.drop(range(10000, index))
    #print(len(dataset), '-', len(_get_ids(dataset)), '=', len(dataset) - len(_get_ids(dataset)))
    #print(_get_hamming_distances(dataset))
    print(entropy(dataset)[:10])
