import numpy as np


# C4.5

def entropy(dataset):
    _, counts = np.unique(dataset[:, -1], return_counts=True)
    ent = 0
    for count in counts:
        pk = count / len(dataset)
        ent += -pk * np.log2(pk)

    return ent


def calculate_gain(dataset, attr_idx):
    total_samples = len(dataset)
    unique_values, counts = np.unique(dataset[:, attr_idx], return_counts=True)
    dataset_entropy = entropy(dataset)
    subset_entropies = 0
    for value in unique_values:
        subset = dataset[dataset[:, attr_idx] == value]
        subset_entropy = entropy(subset)
        subset_entropies += (len(subset) / total_samples) * subset_entropy

    return dataset_entropy - subset_entropies


def gain_ratio_categorical(dataset, attr_idx):
    gain = calculate_gain(dataset, attr_idx)

    total_samples = len(dataset)
    unique_values, counts = np.unique(dataset[:, attr_idx], return_counts=True)
    intrinsic_values = 0
    for value in unique_values:
        subset = dataset[dataset[:, attr_idx] == value]
        intrinsic_values += (len(subset) / total_samples) * np.log2(len(subset) / total_samples)

    if gain == 0 or intrinsic_values == 0:
        return 0
    intrinsic_values = -intrinsic_values
    return gain / intrinsic_values


def gain_continuous(dataset, attr_idx):
    sorted_dataset = dataset[np.argsort(dataset[:, attr_idx])]
    thresholds = [(float(sorted_dataset[i, attr_idx]) + float(sorted_dataset[i + 1, attr_idx])) / 2
                  for i in range(len(sorted_dataset) - 1)]

    max_gain = -float("inf")
    best_threshold = None

    for threshold in thresholds:
        dataset_before = sorted_dataset[sorted_dataset[:, attr_idx].astype(float) <= threshold]
        dataset_after = sorted_dataset[sorted_dataset[:, attr_idx].astype(float) > threshold]
        gain = entropy(sorted_dataset) - (
                (len(dataset_before) / len(sorted_dataset)) * entropy(dataset_before) +
                (len(dataset_after) / len(sorted_dataset)) * entropy(dataset_after)
        )

        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold

    return max_gain, best_threshold


def gain_ratio_continuous(dataset, attr_idx):
    gain, t = gain_continuous(dataset, attr_idx)

    total_samples = len(dataset)
    subset_l = dataset[dataset[:, attr_idx].astype(float) <= t]
    subset_r = dataset[dataset[:, attr_idx].astype(float) > t]
    intrinsic_values_l = 0 if (len(subset_l) == 0 or total_samples == 0) \
        else ((len(subset_l) / total_samples) * np.log2(len(subset_l) / total_samples))
    intrinsic_values_r = 0 if len(subset_r) == 0 or total_samples == 0 \
        else (len(subset_r) / total_samples) * np.log2(len(subset_r) / total_samples)
    intrinsic_values = -(intrinsic_values_l + intrinsic_values_r)

    if gain == 0 or intrinsic_values == 0:
        return 0

    return gain / intrinsic_values


# CART

def gini(dataset):
    _, counts = np.unique(dataset[:, -1], return_counts=True)
    g_sum = 0
    for count in counts:
        pk = count / len(dataset)
        g_sum += pk * pk

    return 1 - g_sum


def gini_index_categorical(dataset, attr_idx):
    total_samples = len(dataset)
    unique_values, counts = np.unique(dataset[:, attr_idx], return_counts=True)
    subset_ginis = 0
    for value in unique_values:
        subset = dataset[dataset[:, attr_idx] == value]
        subset_gini = gini(subset)
        subset_ginis += (len(subset) / total_samples) * subset_gini

    return subset_ginis


def gini_index_continuous(dataset, attr_idx):
    sorted_dataset = dataset[np.argsort(dataset[:, attr_idx])]
    thresholds = [(float(sorted_dataset[i, attr_idx]) + float(sorted_dataset[i + 1, attr_idx])) / 2
                  for i in range(len(sorted_dataset) - 1)]

    min_gini = float("inf")
    best_threshold = None

    for threshold in thresholds:
        dataset_before = sorted_dataset[sorted_dataset[:, attr_idx].astype(float) <= threshold]
        dataset_after = sorted_dataset[sorted_dataset[:, attr_idx].astype(float) > threshold]
        gain = ((len(dataset_before) / len(sorted_dataset)) * gini(dataset_before) +
                (len(dataset_after) / len(sorted_dataset)) * gini(dataset_after))

        if gain < min_gini:
            min_gini = gain
            best_threshold = threshold

    return min_gini, best_threshold
