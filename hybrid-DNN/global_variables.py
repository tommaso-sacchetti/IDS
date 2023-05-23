import os
from pathlib import Path

DEBUG = False

# Data loading

mod_path = Path(__file__).parent
raw_relative_path = "../data/raw.csv"
raw_data_path = (mod_path / raw_relative_path).resolve()

dataset_name = "CONTINOUS_CHANGE__MASQUERADE__v14"
attack_relative_path = "../data/" + dataset_name + ".csv"
attack_data_path = (mod_path / attack_relative_path).resolve()

# filtering

CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

whitelist_filename = "rules/whitelist.txt"
whitelist_file = os.path.join(CUR_PATH, whitelist_filename)

periods_filename = "rules/periods.npy"
periods_file = os.path.join(CUR_PATH, periods_filename)

# features
hamming_filename = "pre-process/hamming_"
hamming_filename = hamming_filename + dataset_name + ".npy"
hamming_file = os.path.join(CUR_PATH, hamming_filename)

entropy_filename = "pre-process/entropy_"
entropy_filename = entropy_filename + dataset_name + ".npy"
entropy_file = os.path.join(CUR_PATH, entropy_filename)

b1 = 0
b2 = 1

# test

test_name = dataset_name + "_VALIDATION.csv"
test_data_path = (mod_path / test_name).resolve()
