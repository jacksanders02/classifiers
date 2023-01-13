import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt


def get_misclassified(points, w, target_class):
    """
    Finds out which points would be misclassified if w was used as a decision b
    boundary
    :param points: The points to attempt to classify
    :param w: The weights for the current iteration
    :param target_class: The target class for this set of points
    :return: A set of misclassified points
    """
    z = points.dot(w[:-1].T) + w[-1]
    z[z > 0] = 1
    z[z == 0] = 3  # A secret third class
    z[z < 0] = 2
    return zip(points[np.where(z != target_class)], [target_class for _ in
                                                     range(points.shape[0])])


def run_iteration(X, w, learning_rate):
    """
    Runs a single iteration of the perceptron learning algorithm
    :param X: The dataset to learn a decision boundary for
    :param w_current: The current weights
    :param learning_rate: The learning rate
    :return: The new weights for the decision boundary
    """
    w_new = np.array([i for i in w])
    Y = []
    for label in range(X.shape[0]):
        label += 1
        Y.extend(get_misclassified(X[label-1], w, label))
    for point, label in Y:
        modifier = -3 + 2 * label  # -1 for 1, 1 for 2
        w_new -= learning_rate * modifier * np.append(point, 1)

    return w_new


# Generate a linearly separable dataset with two classes
separable = False
min_x = -2
max_x = 2

min_y = -5
max_y = 5
while not separable:
    samples = dts.make_classification(n_samples=100, n_features=2,
                                      n_redundant=0, n_informative=1,
                                      n_clusters_per_class=1, flip_y=-1)
    red = samples[0][samples[1] == 0]
    blue = samples[0][samples[1] == 1]
    separable = any([red[:, k].max() < blue[:, k].min() or red[:,
                                                           k].min() > blue[:,
                                                                      k].max()
                     for k in range(2)])

    min_x = np.min(samples[0][:, 0])
    max_x = np.max(samples[0][:, 0])

    min_y = np.min(samples[0][:, 1])
    max_y = np.max(samples[0][:, 1])

plt.plot(red[:, 0], red[:, 1], 'r.')
plt.plot(blue[:, 0], blue[:, 1], 'b.')

X = np.array([red, blue])

# Run the perceptron learning algorithm
w = np.array([1, 1, -0.5])
for i in range(1000):
    w_old = np.array([i for i in w])
    w = run_iteration(X, w, 0.5)
    if (w == w_old).all():
        print(f"Terminated after {i} iteration(s)")
        break

# Display the generated decision boundary
x = np.linspace(min_x, max_x, 100)
y = -(x * w[0] + w[2]) / w[1]

x = x[np.where(np.logical_and(y >= min_y, y <= max_y))]
y = y[np.logical_and(y >= min_y, y <= max_y)]
plt.plot(x, y)

plt.show()
