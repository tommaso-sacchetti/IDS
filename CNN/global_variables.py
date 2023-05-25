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

# test

test_name = dataset_name + "_VALIDATION.csv"
test_data_path = (mod_path / test_name).resolve()
