import argparse
from argparse import ArgumentParser
import numpy as np
import data_helpers

parser = ArgumentParser()
parser.add_argument("file", type=str, help="the name of your dataset")
parser.add_argument("-n", "--neighbours", type=int, default="1",
                    help="number of neighbours for the k nearest neighbour algorithm")
parser.add_argument("-d", "--distance", type=str, choices=["euclidian", "manhattan"], default="euclidian")
parser.add_argument("-c", "--condensed", action=argparse.BooleanOptionalAction,
                    help="use this flag to condense the training dataset with the Hart Algorithm")
parser.add_argument("-w", "--weighted", action=argparse.BooleanOptionalAction,
                    help="use this flag to give each neighbour a weight of 1/d, where d is the distance to the "
                         "neighbour")
parser.add_argument("-s", "--separated", action=argparse.BooleanOptionalAction,
                    help="use this flag if the training and testing data are in separate files, suffixed with _train "
                         "and _test, respectively")

args = parser.parse_args()

DATASET = args.file
NEIGHBOURS = args.neighbours
DISTANCE = args.distance
CONDENSED = args.condensed
WEIGHTED = args.weighted
SEPARATED = args.separated

# Load dataset
if SEPARATED:
    X = data_helpers.loadfile(DATASET, "_train")
    test_data = data_helpers.loadfile(DATASET, "_test")
else:
    X, test_data = data_helpers.split_data(data_helpers.loadfile(DATASET))


# Takes a point and an array. Returns the array with the manhattan distance per row appended to the end of each row
def manhattan(p1, ps):
    return np.column_stack((ps, np.sum(abs(p1 - ps), axis=1)))


def nearest_neighbours(test, dataset, n):
    distance_array = manhattan(test, dataset)

    # Return array of all indices if dataset is too small to use argpartition on
    if dataset.shape[0] < n + 1:
        return np.arange(0, dataset.shape[0])
    else:
        return np.argpartition(distance_array[:, -1], n)[:n]


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


# Find class labels and set up training and testing arrays
labels = np.unique(X[:, -1])

if CONDENSED:
    X = hart(X)

num = 0
for t in test_data:
    nn = nearest_neighbours(t, X, NEIGHBOURS)
    # Cast to int for use in bincount
    nearest_classes = (nn[:, -1] if nn.ndim > 1 else np.array([nn[-1]])).astype(int)
    if X[np.argmax(np.bincount(nearest_classes))][-1] == t[-1]:
        num += 1

percent_right = num / len(test_data) * 100

print(f'\nClassified {round(percent_right, 2)}% of the testing data correctly with {NEIGHBOURS} nearest neighbours.'
      f' (To 2 D.P.)')
