from pathlib import Path

DEBUG = False
LIMITED_RESOURCES = True

# builds the path for every dataset name in the list of possible datasets
def dataset_path_loader(dataset_names: list):
    paths = list()
    mod_path = Path(__file__).parent
    for name in dataset_names:
        name = "../data/" + name
        paths.append((mod_path / name).resolve())
    return paths

# Data loading

# load a mock dataset for light computation for testing
mod_path = Path(__file__).parent
single_dataset_name = "CONTINOUS_CHANGE__MASQUERADE__v14"
attack_relative_path = "../data/" + single_dataset_name + ".csv"
attack_data_path = (mod_path / attack_relative_path).resolve()

# name of the datasets divided by attack category
dataset_base_names = [
    "continous_change_masquerade",
    "drop",
    "full_replayed",
    "fuzzed_masquerade",
    "injection",
    "replayed_masquerade",
]

# create the name.csv, name_TRAIN.csv and name_TEST.csv dataset names 
# TRAIN and TEST might be of no use, depending, on the future development
# the idea is to split them in different files 
dataset_file_names = [name + '.csv' for name in dataset_base_names]
dataset_train_names = [name + '_TRAIN.csv' for name in dataset_base_names]
dataset_test_names = [name + '_TEST.csv' for name in dataset_base_names]

dataset_full_files = dataset_path_loader(dataset_file_names)
dataset_train_files = dataset_path_loader(dataset_train_names)
dataset_test_files = dataset_path_loader(dataset_test_names)


# dataset features

attack_classes = ["benign"]
attack_classes.extend(dataset_base_names)


if __name__ == '__main__':
    print(dataset_base_names,'\n', dataset_train_names,'\n', dataset_test_names)
    print('\n\n', attack_classes)