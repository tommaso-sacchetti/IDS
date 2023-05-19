import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset_loader
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

CUR_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

DEBUG = False

############################################################
####                RULE-BASED FILTERING                ####
############################################################


def initialize_rules(dataset: pd.DataFrame) -> None:
    # remove all the attack flagged elements from dataframe to initialize clean rules
    dataset = dataset.loc[dataset["flag"] == 0]
    _add_to_whitelist(dataset)
    _store_periods(dataset)


def filter(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_whitelist, id_blacklist = _filter_blacklisted_id(dataset)
    period_whitelist, period_blacklist = _check_dataset_time_intervals(id_whitelist)
    dlc_whitelist, dlc_blacklist = _check_dlc(period_whitelist)

    return dlc_whitelist, id_blacklist, period_blacklist, dlc_blacklist


############################################################
####               ID BLACKLIST FILTERING               ####
############################################################

whitelist_filename = "rules/whitelist.txt"
whitelist_file = os.path.join(CUR_PATH, whitelist_filename)


def _add_to_whitelist(whitelisted_dataset: pd.DataFrame) -> None:
    whitelisted_ids = whitelisted_dataset["id"].unique()
    if os.path.exists(whitelist_file) and os.stat(whitelist_file).st_size != 0:
        # ugly but it works
        with open(whitelist_file, "r+") as f:
            old_whitelisted = []
            for line in f:
                old_whitelisted.append(line.strip())
            new_whitelisted = list(set(whitelisted_ids) - set(old_whitelisted))
            if new_whitelisted.len() != 0:
                for id in (pbar := tqdm(new_whitelisted)):
                    pbar.set_description(f"Adding ID to whitelist: {id}")
                    f.write(str(id) + "\n")
    else:
        with open(whitelist_file, "w") as f:
            for id in (pbar := tqdm(whitelisted_ids)):
                pbar.set_description("whitelisting IDs")
                f.write(str(id) + "\n")


def _filter_blacklisted_id(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(whitelist_file) and os.stat(whitelist_file).st_size != 0:
        with open(whitelist_file, "r") as f:
            whitelisted = []
            for line in (pbar := tqdm(f.readlines())):
                pbar.set_description("filtering blacklisted IDs")
                whitelisted.append(line.strip())
        blacklisted_dataset = dataset.loc[-dataset["id"].isin(whitelisted)]
        whitelisted_dataset = dataset.loc[dataset["id"].isin(whitelisted)]
        # debug prints
        if DEBUG:
            print(
                f"Dataset shape: {dataset.shape}; whitelisted_dataset shape: {whitelisted_dataset.shape}; blacklisted_dataset shape: {blacklisted_dataset.size}"  # noqa: E501
            )
            print(
                f"IDs in whitelist: {len(whitelisted_dataset['id'].unique())}; IDs in blacklist: {len(blacklisted_dataset['id'].unique())}"  # noqa: E501
            )
            print("Blacklisted IDs:", blacklisted_dataset["id"].unique())
        return whitelisted_dataset, blacklisted_dataset
    else: 
        print("Error: no whitelist available. Returning full dataset")
        return dataset, pd.DataFrame()


############################################################
####                    TIME INTERVAL                   ####
############################################################

THRESHOLD = 20
K = 5
periods_filename = "rules/periods.npy"
periods_file = os.path.join(CUR_PATH, periods_filename)


def _store_periods(dataset: pd.DataFrame) -> dict:
    # TODO: STORE ONLY RANGE_MIN AND RANGE_MAX, dict might be good (not the periods, waste of memory)  # noqa: E501
    id_periods = dict()
    if os.path.exists(periods_file):
        id_periods = np.load(periods_file, allow_pickle="TRUE").item()
    for id in (pbar := tqdm(dataset["id"].unique())):
        # TODO: update the periods if changed instead of ignoring
        # currently saved id periods
        pbar.set_description("saving IDs periods")
        if id not in id_periods.keys():
            id_packets = dataset.loc[dataset["id"] == id]
            times_of_arrival = id_packets["time"].to_numpy()
            periods = np.diff(times_of_arrival)
            # inserd ID only if periodic
            if periods.size != 0:
                if _check_periodicity(periods, THRESHOLD):
                    id_periods[id] = periods
    # Save periods of periodic IDs in file
    np.save(periods_file, id_periods)
    return id_periods


def _check_periodicity(periods: np.array, percentual_threshold: int) -> bool:
    avg = periods.mean()
    sigma = periods.std()
    coefficient = sigma / avg
    percentual_threshold = percentual_threshold / 100
    return coefficient < percentual_threshold


def _compute_range(k: int, periods: np.array) -> Tuple[float, float]:
    avg = periods.mean()
    sigma = periods.std()
    range_min = avg - k * sigma
    range_max = avg + k * sigma
    return range_min, range_max


def _check_dataset_time_intervals(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(periods_file) and os.stat(periods_file).st_size != 0:
        id_periods = dict()
        id_periods = np.load(periods_file, allow_pickle="TRUE").item()
        blacklisted_dataset = pd.DataFrame()
        whitelisted_dataset = pd.DataFrame()
        for id in (pbar := tqdm(dataset["id"].unique())):
            pbar.set_description("Checking packet intervals")
            if id in id_periods:
                # compute the periods
                id_packets = dataset.loc[dataset["id"] == id]
                times_of_arrival = id_packets["time"].to_numpy()
                periods = np.diff(times_of_arrival)
                range_min, range_max = _compute_range(K, id_periods[id])
                df = dataset.loc[dataset["id"] == id]
                for index, period in enumerate(periods):
                    if period > range_max or period < range_min:
                        # since first row with id of df is not considered in the deltas
                        # (no way to compute it) it's ignored and the +1 is added
                        if DEBUG:
                            print(
                                f"ID: {id}, MIN: {range_min}, MAX: {range_max}, PERIOD {period}"  # noqa: E501
                            )
                        row = df.iloc[[index]]
                        blacklisted_dataset = pd.concat([blacklisted_dataset, row])
            else:
                # id is not periodic, no way to check it's correct periodicity
                if DEBUG:
                    print(id, " is not periodic")
                continue
    else:
        print("Error: no periods file available. Returning full dataset")
        return dataset, pd.DataFrame()
    if DEBUG:
        print("\n DATASET: ", blacklisted_dataset, "\n\n")
        print(whitelisted_dataset.size, blacklisted_dataset.size)
    whitelisted_dataset = (
        pd.merge(dataset, blacklisted_dataset, indicator=True, how="outer")
        .query('_merge=="left_only"')
        .drop("_merge", axis=1)
    )
    return whitelisted_dataset, blacklisted_dataset


def _plot_periods(id: str, dataset: pd.DataFrame) -> None:
    # debug purpose method
    id_packets = dataset.loc[dataset["id"] == id]
    times_of_arrival = id_packets["time"].to_numpy()
    periods = np.diff(times_of_arrival)
    plt.hist(periods, color="lightgreen", ec="black", bins=100)
    r_min, r_max = _compute_range(K, periods)
    plt.axvline(r_min, 0, 1, label="min")
    plt.axvline(r_max, 0, 1, label="max")
    plt.show()


############################################################
####                      DLC CHECK                     ####
############################################################


def _check_dlc(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    whitelisted_dataset = pd.DataFrame()
    blacklisted_dataset = pd.DataFrame()
    # using numpy array for faster computing time
    dlc = dataset["dlc"].to_numpy()
    payload = dataset["payload"].to_numpy()
    print("Checking data length code...")

    def func(x):
        return len(x) / 8

    payload = np.vectorize(func)(payload)
    diff = np.subtract(dlc, payload)
    indexes = np.argwhere(diff).flatten()
    blacklisted_dataset = dataset.iloc[indexes]
    if len(blacklisted_dataset) != 0:
        whitelisted_dataset = (
            pd.merge(dataset, blacklisted_dataset, indicator=True, how="outer")
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )
    else:
        whitelisted_dataset = dataset
    return whitelisted_dataset, blacklisted_dataset


############################################################
####                      TESTING                       ####
############################################################

if __name__ == "__main__":
    dataset = dataset_loader.get_dataset()
    initialize_rules(dataset)
    print(filter(dataset))
    # a, b, c, d = filter(dataset)
    # print(len(a), len(b), len(c), len(d))
    # print(len(a) + len(b) + len(c) + len(d), len(dataset))
