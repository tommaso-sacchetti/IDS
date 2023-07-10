import os
import time
import math
import dataset_loader as loader
import pandas as pd
import numpy as np
import dataset_loader
import global_variables as glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# TODO: fix the name, find a way to make it available to every file
def get_features(
    dataset: pd.DataFrame, b1: int, b2: int, dataset_name = glob.dataset_name
) -> pd.DataFrame:
    # TODO: see if hamming or entropy are stored and get it from them
    '''
        Extracts the relevant features from the raw dataset
        Arguments:
            dataset: a CAN dataset (DataFrame)
            b1: integer representing first important byte to consider
            b2: integer representing second important byte to consider
            dataset_name: str, name of dataset, for file naming purpose only

        Returns:
            DataFrame containing the features with the following columns:
                ID | HAMMING | ENTROPY | B1(int) | B2(int) | FLAG 
    '''
    ids = _get_ids(dataset)
    hamming = _get_hamming_distances(dataset, dataset_name)
    entropy = _entropy(dataset, dataset_name)
    importance_bytes = _get_bytes(dataset, b1, b2)
    flags = dataset["flag"].to_numpy()
    features = np.column_stack((ids, hamming, entropy, importance_bytes, flags))
    features_name = [
        "id",
        "hamming",
        "entropy",
        "byte" + str(b1) + " (int)",
        "byte" + str(b2) + " (int)",
        "flags",
    ]
    features = pd.DataFrame(features)
    features.columns = features_name
    return features


def _get_ids(dataset: pd.DataFrame) -> np.array:
    ids = dataset["id"].to_numpy()

    def func(x):
        return int(str(x), 16)

    x = np.vectorize(func)(ids)
    return x


def _get_hamming_distances(dataset: pd.DataFrame, dataset_name: str) -> np.array:

    # TODO: try a faster way to compute hamming distances
    if not os.path.exists(glob.hamming_file):
        ids = _get_ids(dataset)
        payloads = dataset["payload"].to_numpy()
        id_payload = list(zip(ids, payloads))
        last_message = dict()
        hamming_distances = np.empty(0, int)
        start_time = time.time()
        for id, payload in (pbar := tqdm(id_payload)):
            pbar.set_description("Computing Hamming distances")
            if id not in last_message:
                hamming_distances = np.append(hamming_distances, 0)
                last_message[id] = payload
            else:
                distance = _hamming(payload, last_message[id])
                hamming_distances = np.append(hamming_distances, distance)
                last_message[id] = payload
        execution_time = time.time() - start_time
        print(
            f"Total excecution time for hamming distances: {round(execution_time, 3)}s"
        )
    else:
        hamming_distances = np.load(glob.hamming_file, allow_pickle="TRUE").tolist()
    return hamming_distances


def _hamming(s1: str, s2: str) -> int:
    assert len(s1) == len(s2)
    return bin(int(s1, 2) ^ int(s2, 2)).count("1")


def _entropy(dataset: pd.DataFrame, dataset_name: str) -> np.array:
    # TODO: check if entropy is correct
    if not os.path.exists(glob.entropy_file):
        distribution = np.zeros(2**8)
        # calculation of the distribution
        payloads = dataset["payload"].to_numpy()
        for payload in (pbar := tqdm(payloads)):
            pbar.set_description("Calculating single bytes distribution (entropy)")
            single_bytes = _split_string(payload)
            for byte in single_bytes:
                distribution[int(byte, 2)] += 1
        distribution = distribution / np.sum(distribution)
        plt.hist(distribution, color="lightgreen", ec="black", bins=500)
        plt.title("byte value distribution histogram")
        plt.show()
        # calculation of entropy
        entropy = np.empty(0, int)
        for payload in (pbar := tqdm(payloads)):
            pbar.set_description("Computing entropy")
            single_bytes = _split_string(payload)
            sum = 0
            for byte in single_bytes:
                integer = int(byte, 2)
                pk = distribution[integer]
                sum += pk * math.log(pk)
            entropy = np.append(entropy, sum)
        np.save(glob.entropy_file, entropy)
    else:
        entropy = np.load(glob.entropy_file, allow_pickle="TRUE").tolist()
    return entropy


def _split_string(s: str) -> list:
    chunks, chunk_size = len(s), 8
    return [s[i : i + chunk_size] for i in range(0, chunks, chunk_size)]


def _split_and_fill(s: str) -> list:
    """
    splits the payload in single bytes, converts them in integers.
    adds padding of value -1 for every missing byte from the 8 possible
    """
    chunks, chunk_size = len(s), 8
    payload = [int(s[i : i + chunk_size], 2) for i in range(0, chunks, chunk_size)]
    while len(payload) < 8:
        payload.append(-1)
    return payload


def _get_bytes(dataset: pd.DataFrame, b1: int, b2: int) -> np.array:
    payloads = dataset["payload"].to_numpy()
    importance_bytes = np.array(
        [
            (_split_and_fill(payload)[b1], _split_and_fill(payload)[b2])
            for payload in payloads
        ]
    )
    return importance_bytes


if __name__ == "__main__":
    dataset = dataset_loader.get_dataset(
        dataset_name=glob.dataset_name+".csv"
    )
    dataset = dataset[:1000]
    x = get_features(dataset, 0, 1)
    can = loader.CANDataset(x)
    print(get_features(dataset, 0, 1))
