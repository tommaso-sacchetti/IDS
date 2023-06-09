import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset_loader
import global_variables as glob
from tqdm import tqdm
from typing import Tuple


############################################################
####                RULE-BASED FILTERING                ####
############################################################


def initialize_rules(dataset: pd.DataFrame) -> None:
    # remove all the attack flagged elements from dataframe to initialize clean rules
    """
    Initializes the rules of the filter.
    The rules consist in the ID whitelist stored in whitelist.txt
    and the periodicity of the IDs stored in periods.npy

    Arguments:
        dataset: the dataset from which to store the rules

    Returns:
        None
    """
    dataset = dataset.loc[dataset["flag"] == 0]
    _add_to_whitelist(dataset)
    _store_periods(dataset)


def filter(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filters the dataset following the rules: ID -> periodicity -> DLC

    Arguments:
        dataset: the dataset to filter

    Returns
        whitelist: the dataframe with all the whitelisted messages
        id_blacklist: the dataframe with messages blacklisted for their ID
        period_blacklist: the dataframe with messages blacklisted for their periodicity
        dlc_blacklist: the dataframe with messages blacklisted for their DLC (wrong DLC)
    """
    id_whitelist, id_blacklist = _filter_blacklisted_id(dataset)
    period_whitelist, period_blacklist = _check_dataset_time_intervals(id_whitelist)
    dlc_whitelist, dlc_blacklist = _check_dlc(period_whitelist)

    return dlc_whitelist, id_blacklist, period_blacklist, dlc_blacklist


############################################################
####               ID BLACKLIST FILTERING               ####
############################################################


def _add_to_whitelist(whitelisted_dataset: pd.DataFrame) -> None:
    """
    Adds to whitelist every ID contained in the dataset inserted in input
    """
    whitelisted_ids = whitelisted_dataset["id"].unique()
    if (
        os.path.exists(glob.whitelist_file)
        and os.stat(glob.whitelist_file).st_size != 0
    ):
        # ugly but it works
        with open(glob.whitelist_file, "r+") as f:
            old_whitelisted = []
            for line in f:
                old_whitelisted.append(line.strip())
            new_whitelisted = list(set(whitelisted_ids) - set(old_whitelisted))
            if len(new_whitelisted) != 0:
                for id in (pbar := tqdm(new_whitelisted)):
                    pbar.set_description(f"Adding ID to whitelist: {id}")
                    f.write(str(id) + "\n")
    else:
        with open(glob.whitelist_file, "w") as f:
            for id in (pbar := tqdm(whitelisted_ids)):
                pbar.set_description("whitelisting IDs")
                f.write(str(id) + "\n")


def _filter_blacklisted_id(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters by ID every message in the dataset
    If the ID of the message is not present in the whitelist this function blacklists it

    Returns:
        (whitelist, blacklist): tuple of DataFrames, whitelisted and blacklisted

    """
    if (
        os.path.exists(glob.whitelist_file)
        and os.stat(glob.whitelist_file).st_size != 0
    ):
        with open(glob.whitelist_file, "r") as f:
            whitelisted = []
            for line in (pbar := tqdm(f.readlines())):
                pbar.set_description("filtering blacklisted IDs")
                whitelisted.append(line.strip())
        blacklisted_dataset = dataset.loc[-dataset["id"].isin(whitelisted)]
        whitelisted_dataset = dataset.loc[dataset["id"].isin(whitelisted)]
        # debug prints
        if glob.DEBUG:
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


def _store_periods(dataset: pd.DataFrame) -> dict:
    # TODO: STORE ONLY RANGE_MIN AND RANGE_MAX, dict might be good (not the periods, waste of memory)  # noqa: E501
    """
    Store the periods of every single ID
    Periods are stored in the file rules/periods.npy

    The stored data structure consists in a dictionary, where for every ID
    there is a corresponding numpy array containing all the differences
    between a line and the other hence all the periods from message to message
    of the same ID

    Returns:
        id_periods: dict containing the periods for every periodic ID
    """

    # To store all periods and dinamically update it, it might be better 
    # to initialize only once with a dataset containing all the messages of a 
    # given id
    id_periods = dict()
    if os.path.exists(glob.periods_file):
        id_periods = np.load(glob.periods_file, allow_pickle="TRUE").item()
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
    np.save(glob.periods_file, id_periods)
    return id_periods


def _check_periodicity(periods: np.array, percentual_threshold: int) -> bool:
    """
    Checks the periodicity of a given numpy array of periods (for an ID)
    It checks according to a coefficient which is std_deviation/average

    Arguments:
        periods: a numpy array containing all the periods between messages
        percentual_threshold: the coefficient threshold under which
            the message is considered periodic

    Returns:
        boolean: true if periodic, false otherwise
    """
    avg = periods.mean()
    sigma = periods.std()
    coefficient = sigma / avg
    percentual_threshold = percentual_threshold / 100
    return coefficient < percentual_threshold


def _compute_range(k: int, periods: np.array) -> Tuple[float, float]:
    """
    Computes the range for a np.array to consider an element of the same "family"
    (same ID in this case) periodic.

    Arguments:
        k: hyperparameter indicating the tolerance of the range
        periods: numpy array containing the periods

    Returns:
        range_min: integer, minumun value to be periodic
        range_max: integer, maximum value to be periodic
    """
    avg = periods.mean()
    sigma = periods.std()
    range_min = avg - k * sigma
    range_max = avg + k * sigma
    return range_min, range_max


def _check_dataset_time_intervals(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Actual filter to check if messages are inside the correct periodic range

    Arguments:
        dataset: raw dataset to filter

    Returns:
        whitelisted: DataFrame with the whitelist
        blacklisted: DataFrame with the blacklist

    """
    if os.path.exists(glob.periods_file) and os.stat(glob.periods_file).st_size != 0:
        id_periods = dict()
        id_periods = np.load(glob.periods_file, allow_pickle="TRUE").item()
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
                        if glob.DEBUG:
                            print(
                                f"ID: {id}, MIN: {range_min}, MAX: {range_max}, PERIOD {period}"  # noqa: E501
                            )
                        row = df.iloc[[index]]
                        blacklisted_dataset = pd.concat([blacklisted_dataset, row])
            else:
                # id is not periodic, no way to check it's correct periodicity
                if glob.DEBUG:
                    print(id, " is not periodic")
                continue
    else:
        print("Error: no periods file available. Returning full dataset")
        return dataset, pd.DataFrame()

    if glob.DEBUG:
        print("\n DATASET: ", blacklisted_dataset, "\n\n")
        print(whitelisted_dataset.size, blacklisted_dataset.size)

    if blacklisted_dataset.empty:
        return dataset, pd.DataFrame()

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
    """
    Check the correcteness of the DLC of the messages.
    If the DLC doesn't match the actual number of bytes of the payload
    then the message goes in the blacklist
    """

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

