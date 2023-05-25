import dataset_loader
import pandas as pd
import numpy as np

def get_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Get the features for the dataset
    Features consist in:
        IDs as integers,
        payload as arrays of bytes converted to integer (-1 where byte not present)
        flags: 0 normal message, 1 malicious
    """
    ids = dataset["id"].to_numpy()

    def func(x):
        return int(x, base=16)

    ids = np.vectorize(func)(ids)
    payloads = dataset["payload"].to_numpy()
    payloads = np.array([_split_and_fill(payload) for payload in payloads])
    flags = dataset["flag"].to_numpy()
    features = np.column_stack((ids, payloads, flags))
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
    dataset = dataset_loader.get_dataset()
    features = get_features(dataset)
    print(dataset[:10])
