import os
import time
import math
import dataset_loader as loader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def get_features(
    dataset: pd.DataFrame, b1: int, b2: int, dataset_name: str
) -> pd.DataFrame:
    # TODO: see if hamming or entropy are stored and get it from them
    ids = _get_ids(dataset)
    hamming = _get_hamming_distances(dataset, dataset_name)
    entropy = _entropy(dataset, dataset_name)
    importance_bytes = _get_bytes(dataset, b1, b2)
    flags = dataset["flag"].to_numpy()
    features = np.column_stack((ids, hamming, entropy, importance_bytes, flags))
    return pd.DataFrame(features)


def _get_ids(dataset: pd.DataFrame) -> np.array:
    ids = dataset["id"].to_numpy()

    def func(x):
        return int(str(x), 16)

    x = np.vectorize(func)(ids)
    return x


def _get_hamming_distances(dataset: pd.DataFrame, dataset_name: str) -> np.array:
    hamming_filename = "pre-process/hamming_"
    hamming_filename = hamming_filename + dataset_name + ".npy"
    hamming_file = os.path.join(CUR_PATH, hamming_filename)

    # TODO: try a faster way to compute hamming distances
    if not os.path.exists(hamming_file):
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
        np.save(hamming_file, hamming_distances)
    else:
        hamming_distances = np.load(hamming_file, allow_pickle="TRUE").item()
    return hamming_distances


def _hamming(s1: str, s2: str) -> int:
    assert len(s1) == len(s2)
    return bin(int(s1, 2) ^ int(s2, 2)).count("1")


def _entropy(dataset: pd.DataFrame, dataset_name: str) -> np.array:
    # TODO: check if entropy is correct
    entropy_filename = "pre-process/entropy_"
    entropy_filename = entropy_filename + dataset_name + ".npy"
    entropy_file = os.path.join(CUR_PATH, entropy_filename)
    if not os.path.exists(entropy_file):
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
        plt.set_title("byte value distribution histogram")
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
        np.save(entropy_file, entropy)
    else:
        entropy = np.load(entropy_file, allow_pickle="TRUE").item()
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
    mod_path = Path(__file__).parent
    relative_path = "../data/raw.csv"
    data_path = (mod_path / relative_path).resolve()
    colnames = ["time", "can", "id", "dlc", "payload"]
    dataset = pd.read_csv(data_path, names=colnames, header=None, nrows=100)
    x = get_features(dataset, 0, 1)
    can = loader.CANDataset(x)
    # index = dataset.index[-1]
    # dataset = dataset.drop(range(100, index))
    # print(len(dataset), '-', len(_get_ids(dataset)), '=', len(dataset) - len(_get_ids(dataset)))
    # print(_get_hamming_distances(dataset))
    # print(entropy(dataset)[:10])
    print(get_features(dataset, 0, 1))
