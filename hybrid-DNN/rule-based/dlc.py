import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

FILENAME = 'whitelist.txt'
CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file = (os.path.join(CUR_PATH, FILENAME))

def check_dlc(dataset):
    whitelisted_dataset = pd.DataFrame()
    blacklisted_dataset = pd.DataFrame()
    for index, packet in tqdm(dataset.iterrows()):
        dlc = int(packet['dlc'])
        payload = packet['payload']
        length = len(payload) / 8
        if dlc != length:
            blacklisted_dataset = pd.concat([blacklisted_dataset, dataset.iloc[[index]]])
    if len(blacklisted_dataset) != 0:
        whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return whitelisted_dataset, blacklisted_dataset

def check_dlc2(dataset):
    whitelisted_dataset = pd.DataFrame()
    blacklisted_dataset = pd.DataFrame()
    # using numpy array for faster computing time
    dlc = dataset['dlc'].to_numpy()
    payload = dataset['payload'].to_numpy()
    def func(x):
        print("Checking data length code...")
        return len(x)/8
    payload = np.vectorize(func)(payload)
    diff = np.subtract(dlc, payload)
    indexes = np.argwhere(diff).flatten()
    blacklisted_dataset = dataset.iloc[indexes]
    if len(blacklisted_dataset) != 0:
        whitelisted_dataset = pd.merge(dataset,blacklisted_dataset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return whitelisted_dataset, blacklisted_dataset

if __name__ == '__main__':
    mod_path = Path(__file__).parent
    relative_path = '../../data/raw.csv'
    DATA_PATH = (mod_path / relative_path).resolve()  # 'data/raw.csv'
    colnames = ['time', 'can', 'id', 'dlc', 'payload']
    dataset = pd.read_csv(DATA_PATH, names=colnames, header=None)
    # creating mock fake row with wrong dlc to test
    time = np.float64(0)
    dlc = np.int64(1)
    test_row = pd.DataFrame({'time': time, 'can': 'can0', 'id': '0', 'dlc': dlc, 'payload': '0000000000000000'}, index=[0])
    dataset = pd.concat([test_row, dataset.loc[:]]).reset_index(drop=True)
    dataset = pd.concat([test_row, dataset.loc[:]]).reset_index(drop=True)

    index = dataset.index[-1]
    #dataset = dataset.drop(range(50, index))
    white, black = check_dlc2(dataset)
    print(black)