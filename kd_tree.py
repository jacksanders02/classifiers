import math

import numpy as np

class KDNode(object):
    def __init__(self, node, left, right, axis, dim, id):
        self.node = node
        self.parent = None
        self.side = None

        self.left = left
        if self.left is not None:
            self.left.parent = self
            self.left.side = "l"

        self.right = right
        if self.right is not None:
            self.right.parent = self
            self.right.side = "r"

        self.axis = axis
        self.dim = dim
        self.id = id

    def __str__(self):
        return f'{self.left} | {self.node} | {self.right}'

    def get_nth_parent(self, n):
        if n == 0 or self.parent is None:
            return self
        else:
            return self.parent.get_nth_parent(n - 1)

    def check_node(self, t, current_bests, checked):
        checked.add(self.id)

        node_dist = (manhattan_1d(t[:-1], self.node[:-1]), self.node[-1])

        #print(node_dist, current_bests)
        if node_dist[0] < current_bests[-1][0]:
            current_bests[-1] = node_dist

            # Sort based on distance
            current_bests = current_bests[current_bests[:, -2].argsort()]

        for c in "l", "r":
            child = self.left if c == "l" else self.right
            if child is None or child.id in checked:
                continue

            checked.add(child.id)

            test_col = t[self.axis]

            #print(abs(test_col - child.node[self.axis]) , current_bests[-1][0])
            if math.pow(abs(test_col - self.node[self.axis]), 4) < current_bests[-1][0]:
                current_bests = child.check_node(t, current_bests, checked)

        return current_bests


def build_tree(data: np.array, depth: int = 0, new_id=0):
    if data.size == 0:
        return None

    # Get axis based on dimension
    sort_col = depth % (len(data[0]) - 1)

    # Sort data based on current sort column
    data = data[data[:, sort_col].argsort()]
    median = len(data) // 2

    return KDNode(data[median],
                  build_tree(data[:median], depth + 1, new_id * 2 + 1),
                  build_tree(data[median + 1:], depth + 1, new_id * 2 + 2),
                  sort_col,
                  len(data[0]) - 1,
                  new_id)


# Takes two points. Returns the second array with the manhattan distance appended to the end
def manhattan_1d(p1, p2):
    return np.sum(abs(p1 - p2))


def nn_kd_tree(test: np.array, start_node: KDNode, current_bests: np.array):
    last_node = None
    current_node = start_node

    while current_node is not None:
        last_node = current_node
        if test[current_node.axis] < current_node.node[current_node.axis]:
            current_node = current_node.left
        else:
            current_node = current_node.right

    current_node = last_node
    while current_node:
        current_bests = current_node.check_node(test, current_bests, set())
        current_node = current_node.parent

    return current_bests


if __name__ == "__main__":
    i = 1
    while (a := (4 - i) % 8) != 4:
        print(a)
        i += 1
