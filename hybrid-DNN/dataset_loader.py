import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

mod_path = Path(__file__).parent
relative_path = "../data/raw.csv"
data_path = (mod_path / relative_path).resolve()


def get_data_loader(dataset: pd.DataFrame, batch_size: int) -> DataLoader:
    ds = CANDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def get_dataset() -> pd.DataFrame:
    dataset = _load_dataset_raw()
    dataset = _add_clean_flag_column(dataset)  # replace with real method for flags
    # dataset = _add_flag_column(dataset)
    return dataset


def _load_dataset_raw() -> pd.DataFrame:
    colnames = ["time", "can", "id", "dlc", "payload"]
    dataset = pd.read_csv(data_path, names=colnames, header=None)
    return dataset


def _add_clean_flag_column(dataset: pd.DataFrame) -> pd.DataFrame:
    flags = np.empty(len(dataset), dtype=str)
    flags[:] = "T"
    dataset["flag"] = flags.tolist()
    return dataset


class CANDataset(Dataset):
    # flag: T: injected message, R: normal message
    def __init__(self, dataset : pd.DataFrame):
        x = dataset.iloc[:, 0:-1].values
        y = dataset.iloc[:, -1]
        y.replace({"T": 1, "R": 0}, inplace=True)
        y = y.values

        # if dataset is in a 2d numpy array
        #x = dataset[:, 0:-1]
        #y = dataset[:, -1]
        #y = y.char.replace(y, "T", "1")
        #y = y.char.replace(y, "R", "0")
        #y = y.astype(int)

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.int32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


if __name__ == "__main__":
    loader = get_data_loader(batch_size=10)
    # test
    for i, (data, labels) in enumerate(loader):
        print(data.shape, labels.shape)
        print(data, labels)
        break
