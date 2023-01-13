import numpy as np
import sklearn.datasets as dts
import matplotlib.pyplot as plt


# Source: https://stackoverflow.com/a/66351218
def linePoints(a=0,b=0,c=0,ref = [-1.,1.]):
    """given a,b,c for straight line as ax+by+c=0,
    return a pair of points based on ref values
    e.g linePoints(-1,1,2) == [(-1.0, -3.0), (1.0, -1.0)]
    """
    if (a==0 and b==0):
        raise Exception("linePoints: a and b cannot both be zero")
    return [(-c/a,p) if b==0 else (p,(-c-a*p)/b) for p in ref]


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
# Source: https://stackoverflow.com/a/47964170
def linearly_separable_set():
    separable = False
    red = []
    blue = []
    while not separable:
        samples = dts.make_classification(n_samples=500, n_features=2,
                                          n_redundant=0, n_informative=1,
                                          n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:,
                                                               k].min() > blue[
                                                                          :,
                                                                          k].max()
                         for k in range(2)])
    return np.array([red, blue])


X = linearly_separable_set()

min_x = np.min(X[:, :, 0])
max_x = np.max(X[:, :, 0])

min_y = np.min(X[:, :, 1])
max_y = np.max(X[:, :, 1])

# Run the perceptron learning algorithm
w = np.random.random(3)
for i in range(1000):
    w_old = np.array([i for i in w])
    w = run_iteration(X, w, 0.5)
    if (w == w_old).all():
        print(f"Terminated after {i} iteration(s). Weights = {w}.T")
        break

plt.ion()

fig, ax = plt.subplots()

plt.xlim([min_x, max_x])
plt.ylim([min_y, max_y])

ax.axline(*linePoints(w[0], w[1], w[2]))

ax.plot(X[0][:, 0], X[0][:, 1], 'r.')
ax.plot(X[1][:, 0], X[1][:, 1], 'b.')

plt.show()
