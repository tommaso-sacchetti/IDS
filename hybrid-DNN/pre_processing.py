from os import path
import csv
from itertools import islice 
import sys
import pandas as pd

DATA_PATH = '../data/raw.csv'
sys.path.append(DATA_PATH)

colnames = ['time', 'can', 'id', 'dlc', 'payload']

# CHIEDI SE ID EXTENDED


#raw = pd.read_csv(DATA_PATH, names=colnames, header=None)

#print(raw['id'].unique())





print(504446987 < 2**29)