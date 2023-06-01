import torch
import pre_processing
import pandas as pd
import global_variables as glob
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader


# get a single dataset given a file name
def get_single_dataset(file) -> pd.DataFrame:
    # TODO: put the dataset file name in a place easy to change
    """
    Get the dataset according to the dataset name put in input
    """
    colnames = ["order", "time", "can", "id", "dlc", "payload", "flag"]
    dataset = pd.read_csv(file, header=0)
    dataset.columns = colnames
    return dataset


# gets dataset composed of all datasets, as concatenation of dataframes
# for binary classification
def get_full_binary_dataset(train=True) -> pd.DataFrame:
    training = glob.dataset_train_files
    testing = glob.dataset_test_files
    files = training if train else testing

    # FOR SW TESTING PURPOSE
    # when the dataset structure and testing is better defined
    # the files to load are gonna be changed
    # potentially to dataset_train_files and dataset_test_files
    files = glob.dataset_full_files

    dataset = pd.DataFrame()
    for file in files:
        next = get_single_dataset(file)
        # TODO: change following lines, for line small samples for testing purpose
        # dataset_list =[get_single_dataset(file).sample() for file in files ]
        # dataset = pd.concat([dataset, next], ignore_index=True)
        dataset = pd.concat([dataset, next.sample(n=50000)], ignore_index=True)
    return dataset


# gets dataset composed of all datasets, as a list of dataframes
# for multiclass classification
def get_full_multiclass_dataset(train=True):
    # TODO: set as definitive the training and testing files
    training = glob.dataset_train_files
    testing = glob.dataset_test_files
    files = training if train else testing

    # FOR SW TESTING PURPOSE
    # when the dataset structure and testing is better defined
    # the files to load are gonna be changed
    # potentially to dataset_train_files and dataset_test_files
    files = glob.dataset_full_files

    # TODO: change following lines, for line small samples for testing purpose
    # dataset_list =[get_single_dataset(file).sample() for file in files ]
    dataset_list = [get_single_dataset(file).sample(n=50000) for file in files]

    return dataset_list


# returns a data_loader given a preprocessed dataset
def get_data_loader(
    is_binary: bool, dataset: pd.DataFrame, batch_size: int
) -> DataLoader:
    data_class = CAN_Binary_Dataset if is_binary else CAN_Multiclass_Dataset
    ds = data_class(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def TEST_DATASET():
    dataset = get_single_dataset(glob.attack_data_path)
    print(dataset.sample(n=5))
    features = pre_processing.get_binary_features(dataset)
    mapping = {0: "no_attack", 1: "attack_name"}
    # dataset["flag"] = dataset["flag"].replace(mapping)
    features.iloc[:, -1] = features.iloc[:, -1].replace(mapping)
    return CAN_Binary_Dataset(features)


class CAN_Binary_Dataset(Dataset):
    # OneHotEncoded flags
    def __init__(self, dataset: pd.DataFrame):
        x = dataset.iloc[:, 0:-1]
        y = dataset.iloc[:, -1]
        x = x.to_numpy()
        y = y.to_numpy()

        self.x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        # unsqueeze(1) to change the shape in [batch_size, 1] (for training)
        self.y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[:, idx, :], self.y_train[idx]


class CAN_Multiclass_Dataset(Dataset):
    # OneHotEncoded flags
    def __init__(self, dataset: pd.DataFrame):
        x = dataset.iloc[:, 0:-1]
        y = dataset.iloc[:, -1]
        x = x.to_numpy()
        ohe = OneHotEncoder()
        y = ohe.fit_transform(pd.DataFrame(y))

        self.x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        self.y_train = torch.tensor(y.toarray(), dtype=torch.float32)#.unsqueeze(1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[:, idx, :], self.y_train[idx]


if __name__ == "__main__":
    TEST_DATASET()

    # dataset = get_full_multiclass_dataset()
    # features = pre_processing.get_multiclass_features(dataset)
    # print(features.loc[features.iloc[:,-1] != 'benign'].sample(n=1000))
    # loader = CAN_Dataset(features)

    """
    if glob.LIMITED_RESOURCES: 
        dataset = get_single_dataset(file=glob.attack_data_path)
    else:
        dataset = get_full_binary_dataset()
    print(len(dataset))
    
    dataset = dataset[:1000]
    dataset = pre_processing.get_features(dataset)
    loader = get_data_loader(dataset, batch_size=10)
    for i, (data, labels) in enumerate(loader):
        print(dataset[:10])
        print(data.shape, labels.shape)
        print(data, labels)
        break

    """