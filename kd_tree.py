import numpy as np


class KDNode(object):
    def __init__(self, loc, left, right):
        self.loc = loc
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left} | {self.loc} | {self.right}'


def build_tree(data: np.array, depth: int = 0):
    if data.size == 0:
        return None

    # Get axis based on dimension
    sort_col = depth % len(data[0])

    # Sort data based on current sort column
    data = data[data[:, sort_col].argsort()]
    median = len(data) // 2

    return KDNode(data[median], build_tree(data[:median], depth + 1), build_tree(data[median + 1:], depth + 1))


if __name__ == "__main__":
    pass
