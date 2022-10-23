import argparse
import time
from argparse import ArgumentParser
import numpy as np
import data_helpers

import kd_tree
import pickle
from pathlib import Path


# Takes a point and an array. Returns the array with the manhattan distance per row appended to the end of each row
def manhattan(p1, ps):
    return np.column_stack((ps, np.sum(abs(p1 - ps), axis=1)))


def nearest_neighbours(test, dataset, n):
    distance_array = manhattan(test, dataset)

    # Return array of all indices if dataset is too small to use argpartition on
    if dataset.shape[0] < n + 1:
        return np.arange(0, dataset.shape[0])
    else:
        return dataset[np.argpartition(distance_array[:, -1], n)[:n]]


def build_kd(dataset):
    start = time.time()
    tree = kd_tree.build_tree(dataset)
    pickle.dump(tree, open(f'data/{DATASET}_KD_TREE.kd', 'wb'))
    print(f'Built and saved KD tree in {round(time.time() - start, 2)} seconds.')
    return tree


def load_kd():
    tree = pickle.load(open(f'data/{DATASET}_KD_TREE.kd', 'rb'))
    print(f"Loaded KD tree for dataset {DATASET} successfully.")
    return tree


# Applies the Hart algorithm on the training dataset to produce a condensed classification dataset
def hart(dataset):
    u = np.array([dataset[0]])
    m = 1
    j = 0
    while m > 0:
        i = 0
        while i < dataset.shape[0]:
            x = dataset[i]
            m = 0

            nn = u[nearest_neighbours(x[:-1], u[:, :-1], NEIGHBOURS)]

            # Cast to int for use in bincount
            nearest_classes = (nn[:, -1] if nn.ndim > 1 else np.array([nn[-1]])).astype(int)
            if u[np.argmax(np.bincount(nearest_classes))][-1] != x[-1]:
                u = np.vstack((u, x))
                dataset = np.delete(dataset, i, axis=0)
                i -= 1  # Account for the removal of an element
                m += 1
            i += 1
        j += 1
    return u


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="the name of your dataset")
    parser.add_argument("-n", "--neighbours", type=int, default="1",
                        help="number of neighbours for the k nearest neighbour algorithm")
    parser.add_argument("-w", "--weighted", action=argparse.BooleanOptionalAction,
                        help="use this flag to give each neighbour a weight of 1/d, where d is the distance to the "
                             "neighbour")
    parser.add_argument("-s", "--separated", action=argparse.BooleanOptionalAction,
                        help="use this flag if the training and testing data are in separate files, suffixed with "
                             "_train and _test, respectively")
    parser.add_argument("-kd", "--kd", action=argparse.BooleanOptionalAction,
                        help="use this flag to use KD trees for a much faster search")

    args = parser.parse_args()

    DATASET = args.file
    NEIGHBOURS = args.neighbours
    WEIGHTED = args.weighted
    SEPARATED = args.separated
    USE_KD = args.kd

    if SEPARATED:
        test_data = data_helpers.loadfile(DATASET, "_test")
    else:
        X, test_data = data_helpers.split_data(data_helpers.loadfile(DATASET))

    # Load dataset
    if USE_KD:
        if Path(f'data/{DATASET}_KD_TREE.kd').exists():
            KD = load_kd()
        else:
            if SEPARATED:
                X = data_helpers.loadfile(DATASET, "_train")
            KD = build_kd(X)
    elif SEPARATED:
        X = data_helpers.loadfile(DATASET, "_train")

    num = 0
    start = time.time()
    i = 1
    for t in test_data:
        starting_bests = np.array([(float('inf'), float('inf')) for _ in range(NEIGHBOURS)])
        nn = kd_tree.nn_kd_tree(t, KD, starting_bests) if USE_KD else nearest_neighbours(t, X, NEIGHBOURS)
        # Cast to int for use in bincount
        nearest_classes = (nn[:, -1] if nn.ndim > 1 else np.array([nn[-1]])).astype(int)

        if WEIGHTED:
            weighted_classes = np.bincount(nearest_classes).astype(float)
            for i in range(len(weighted_classes)):
                weighted_classes[i] = np.sum(1/nn[:, 0][nn[:, -1] == i], axis=0)

            if np.argmax(weighted_classes) == t[-1]:
                num += 1
        else:
            if np.argmax(np.bincount(nearest_classes)) == t[-1]:
                num += 1

        if i % 100 == 0:
            print(i)

        i += 1

    percent_right = num / len(test_data) * 100

    print(f'\nClassified {round(percent_right, 2)}% of the testing data correctly with {NEIGHBOURS} nearest neighbours '
          f'in {round(time.time() - start, 2)} seconds.')
