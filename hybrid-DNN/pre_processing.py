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

def _get_ids(dataset: pd.DataFrame) -> np.array:
    return dataset['id'].to_numpy()

def _get_hamming_distances(dataset: pd.DataFrame) -> np.array:
    # TODO: try a faster way to compute hamming distances
    ids = _get_ids(dataset)
    payloads = dataset['payload'].to_numpy()
    id_payload = list(zip(ids, payloads))
    last_message = dict()
    hamming_distances = np.empty(0, int)
    start_time = time.time()
    for id, payload in (pbar := tqdm(id_payload)):
        pbar.set_description('Computing Hamming distances')
        if id not in last_message:
            hamming_distances = np.append(hamming_distances, 0)
            last_message[id] = payload
        else:
            distance = _hamming(payload, last_message[id])
            hamming_distances = np.append(hamming_distances, distance)
            last_message[id] = payload
    execution_time = time.time() - start_time
    np.save(hamming_file, hamming_distances)
    return hamming_distances

def _hamming(s1: str, s2: str) -> int:
    assert len(s1) == len(s2)
    return bin(int(s1,2)^int(s2,2)).count('1')

def _entropy(dataset: pd.DataFrame) -> np.array: 
    # TODO: check if entropy is correct
    distribution = np.zeros(2**8)
    # calculation of the distribution
    payloads = dataset['payload'].to_numpy()
    for payload in (pbar := tqdm(payloads)):
        pbar.set_description('calculating distribution')
        single_bytes = _split_string(payload)
        for byte in single_bytes:
            distribution[int(byte,2)] += 1
    distribution = distribution / np.sum(distribution)
    # calculation of entropy
    entropy = np.empty(0, int)
    for payload in (pbar := tqdm(payloads)):
        pbar.set_description('calculating entropy')
        single_bytes = _split_string(payload)
        sum = 0
        for byte in single_bytes:
            integer = int(byte,2)
            pk = distribution[integer]
            sum += pk * math.log(pk)
        entropy = np.append(entropy, sum)
    
    plt.hist(entropy, color='lightgreen', ec='black', bins=500)
    plt.show()
    return entropy

def _split_string(s: str) -> list:
    chunks, chunk_size = len(s), 8
    return [s[i:i+chunk_size] for i in range(0, chunks, chunk_size)]

def _split_and_fill(s: str) -> list:
    '''
        splits the payload in single bytes, converts them in integers.
        adds padding of value -1 for every missing byte from the 8 possible
    '''
    chunks, chunk_size = len(s), 8
    payload = [int(s[i:i+chunk_size], 2) for i in range(0, chunks, chunk_size)]
    while(len(payload) < 8):
        payload.append(-1)
    return payload

def _get_bytes(dataset: pd.DataFrame, b1: int, b2: int) -> np.array:
    payloads = dataset['payload'].to_numpy()
    importance_bytes = np.array([(_split_and_fill(payload)[b1], _split_and_fill(payload)[b2]) for payload in payloads])
    return importance_bytes

def get_features(dataset: pd.DataFrame, b1: int, b2: int) -> np.array:
    # TODO: see if hamming or entropy are stored and get it from them
    ids = _get_ids(dataset)
    hamming = _get_hamming_distances(dataset)
    entropy = _entropy(dataset)
    importance_bytes = _get_bytes(dataset, b1, b2)
    features = np.column_stack((ids, hamming, entropy, importance_bytes))

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../data/raw.csv'
    data_path = (mod_path / relative_path).resolve()
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(data_path, names=colnames, header=None, nrows=100)
    #index = dataset.index[-1]
    #dataset = dataset.drop(range(100, index))
    #print(len(dataset), '-', len(_get_ids(dataset)), '=', len(dataset) - len(_get_ids(dataset)))
    #print(_get_hamming_distances(dataset))
    #print(entropy(dataset)[:10])
    get_features(dataset, 0, 1)