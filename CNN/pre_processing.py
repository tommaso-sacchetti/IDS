import dataset_loader
import pandas as pd
import numpy as np
import global_variables as glob
from tqdm import tqdm


# get features for the binary model
def get_binary_features(dataset: pd.DataFrame) -> pd.DataFrame:
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

    return pd.DataFrame(features)


# get features for the multiclass model, flags mapped to the attack classes names
def get_multiclass_features(dataset_list: list()) -> pd.DataFrame:
    """
    Get the features for the dataset (multiclass)
    Features consist in:
        IDs as integers,
        payload as arrays of bytes converted to integer (-1 where byte not present)
        flags: strings, representing the attack class
        -> added mapping for the flags, in this way the flags are mapped to strings
        of the attack type, which are then one hot encoded in the loader class
    """
    features = pd.DataFrame()

    print("\nFEATURE PREPROCESSING")
    print("---------------------\n")
    for index, dataset in (pbar := tqdm(enumerate(dataset_list))):
        pbar.set_description("extracting features from dataset")
        ids = dataset["id"].to_numpy()

        def func(x):
            return int(x, base=16)

        ids = np.vectorize(func)(ids)
        payloads = dataset["payload"].to_numpy()
        payloads = np.array([_split_and_fill(payload) for payload in payloads])
        flags = dataset["flag"].to_numpy()
        dataset = np.column_stack((ids, payloads, flags))
        dataset = pd.DataFrame(dataset)

        # mapping the features
        attack_name = glob.attack_classes[index + 1]
        no_attack = glob.attack_classes[0]
        mapping = {0: no_attack, 1: attack_name}
        dataset.iloc[:, -1] = dataset.iloc[:, -1].replace(mapping)
        features = pd.concat([features, dataset], ignore_index=True)

    return features


# helper function for byte formatting
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
    features = get_binary_features(dataset)
    print(dataset[:10])
