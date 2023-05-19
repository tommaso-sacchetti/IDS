import torch
import pre_processing
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

mod_path = Path(__file__).parent
raw_relative_path = "../data/raw.csv"
attack_relative_path = "../data/CONTINOUS_CHANGE__MASQUERADE__v14.csv"
raw_data_path = (mod_path / raw_relative_path).resolve()
attack_data_path = (mod_path / attack_relative_path).resolve()


def get_data_loader(dataset: pd.DataFrame, batch_size: int) -> DataLoader:
    ds = CANDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def get_dataset() -> pd.DataFrame:
    colnames = ["order", "time", "can", "id", "dlc", "payload", "flag"]
    dataset = pd.read_csv(attack_data_path, header=0)
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
