import pandas as pd
import numpy as np
from pathlib import Path

def get_features(dataset):
    ids = dataset['id'].to_numpy()
    payloads = dataset['payload'].to_numpy()
    payloads = np.array([_split(payload) for payload in payloads])
    features = np.column_stack((ids, payloads))
    print(features)

def _split(s):
    chunks, chunk_size = len(s), 8
    payload = [int(s[i:i+chunk_size], 2) for i in range(0, chunks, chunk_size)]
    while(len(payload) < 8):
        payload.append(-1)
    return payload


if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../data/raw.csv'
    data_path = (mod_path / relative_path).resolve()
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(data_path, names=colnames, header=None, nrows=100)

    get_features(dataset)