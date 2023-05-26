import torch
import pandas as pd
import global_variables as glob
from torch.utils.data import Dataset, DataLoader


def get_data_loader(dataset: pd.DataFrame, batch_size: int) -> DataLoader:
    ds = CAN_Dataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def get_single_dataset(file) -> pd.DataFrame:
    # TODO: put the dataset file name in a place easy to change
    """
    Get the dataset according to the dataset name stored in the global variables file
    """
    colnames = ["order", "time", "can", "id", "dlc", "payload", "flag"]
    dataset = pd.read_csv(file, header=0)
    dataset.columns = colnames
    return dataset


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
        dataset = pd.concat([dataset, next], ignore_index=True)
    return dataset


def get_full_multiclass_dataset(train=True):
    training = glob.dataset_train_files
    testing = glob.dataset_test_files
    files = training if train else testing

    # FOR SW TESTING PURPOSE
    # when the dataset structure and testing is better defined
    # the files to load are gonna be changed
    # potentially to dataset_train_files and dataset_test_files
    files = glob.dataset_full_files

    dataset = pd.DataFrame()
    for index, file in enumerate(files):
        attack_name = glob.attack_classes[index + 1]
        no_attack = glob.attack_classes[0]
        print(no_attack, attack_name)
        df = get_single_dataset(file)
        mapping = {0: no_attack, 1: attack_name}
        df["flag"] = df["flag"].replace(mapping)
        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset


class CAN_Dataset(Dataset):
    # flag: T: injected message, R: normal message
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


if __name__ == "__main__":
    df = get_full_multiclass_dataset(glob.attack_data_path)
    mapping = {0: "no attack", 1: "attack"}
    df["flag"] = df["flag"].replace(mapping)
    print(df.sample(n=50))

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
