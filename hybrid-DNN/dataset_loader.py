import torch
import pre_processing
import pandas as pd
import global_variables as glob
from torch.utils.data import Dataset, DataLoader


def get_data_loader(dataset: pd.DataFrame, batch_size: int) -> DataLoader:
    ds = CANDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def get_dataset() -> pd.DataFrame:
    # TODO: put the dataset file name in a place easy to change 
    """
    Get the dataset according to the dataset name stored in this file
    """
    colnames = ["order", "time", "can", "id", "dlc", "payload", "flag"]
    dataset = pd.read_csv(glob.attack_data_path, header=0)
    dataset.columns = colnames
    return dataset


def get_test_dataset() -> pd.DataFrame:
    # TODO: put the dataset file name in a place easy to change 
    """
    Get the test dataset according to the dataset name stored in this file
    """
    colnames = ["order", "time", "can", "id", "dlc", "payload", "flag"]
    dataset = pd.read_csv(glob.test_data_path, header=0)
    dataset.columns = colnames
    return dataset


class CANDataset(Dataset):
    # flag: T: injected message, R: normal message
    def __init__(self, dataset: pd.DataFrame):
        x = dataset.iloc[:, 0:-1]
        y = dataset.iloc[:, -1]
        x = x.to_numpy()
        y = y.to_numpy()

        self.x_train = torch.tensor(x, dtype=torch.float32)
        # unsqueeze(1) to change the shape in [batch_size, 1] (for training)
        self.y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


if __name__ == "__main__":
    dataset = get_dataset()
    dataset = dataset[:10000]
    dataset = pre_processing.get_features(dataset, 0, 1)
    loader = get_data_loader(dataset, batch_size=10)
    # test
    for i, (data, labels) in enumerate(loader):
        print(data.shape, labels.shape)
        print(data, labels)
        break
