import pandas as pd
import numpy as np
from pathlib import Path


def get_features(dataset: pd.DataFrame) -> pd.DataFrame:
    ids = dataset["id"].to_numpy()

    def func(x):
        return int(x, base=16)

    ids = np.vectorize(func)(ids)
    payloads = dataset["payload"].to_numpy()
    payloads = np.array([_split_and_fill(payload) for payload in payloads])
    features = np.column_stack((ids, payloads))
    # print(features)
    return pd.DataFrame(features)


def _split_and_fill(s):
    """
    splits the payload in single bytes, converts them in integers.
    adds padding of value -1 for every missing byte from the 8 possible
    """
    chunks, chunk_size = len(s), 8
    payload = [int(s[i : i + chunk_size], 2) for i in range(0, chunks, chunk_size)]
    while len(payload) < 8:
        payload.append(-1)
    return payload


if __name__ == "__main__":
    mod_path = Path(__file__).parent
    relative_path = "../data/raw.csv"
    data_path = (mod_path / relative_path).resolve()
    colnames = ["time", "can", "id", "dlc", "payload"]
    dataset = pd.read_csv(data_path, names=colnames, header=None, nrows=100)

    features = get_features(dataset)
