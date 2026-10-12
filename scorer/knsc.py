"""
Quantities shared by all algorithms: the Gaussian kernel, the degree matrix, the
cluster labels obtained from the clustering matrix, and the accuracy metric.

The data matrix X is (m x n) with one sample per column, the clustering matrix H
is (k x n), and the basis matrix F is (n x k).

The algorithms are not defined here. The current best algorithm is in
champion/algorithm.py, and the Evaluator implements each proposal against the
scorer interface.
"""

import numpy as np

# Lower bound for the denominator of a multiplicative update. Without it, a single
# zero entry would produce inf or NaN and invalidate the run. Such a bound is
# standard for multiplicative updates and does not change the fixed points.
EPS = 1e-12


def gausskernel(X, Y, sigma):
    """Gaussian kernel matrix, K[i,j] = exp(-||x_i - y_j||^2 / sigma^2).

    X, Y are (m x n1), (m x n2) with samples in columns.

    The denominator is sigma^2, not 2 sigma^2. The width grids in TASK.md are
    defined for this form. With 2 sigma^2, the grids would cover a different
    range of widths.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    squared_norms_X = np.sum(X ** 2, axis=0)
    squared_norms_Y = np.sum(Y ** 2, axis=0)
    squared_distances = (
        squared_norms_X[:, None] - 2.0 * (X.T @ Y) + squared_norms_Y[None, :]
    )
    # Rounding can make a distance slightly negative where two points coincide.
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return np.exp(-squared_distances / (sigma ** 2))


def degree_matrix(affinity):
    """Diagonal degree matrix D of a non-negative affinity matrix A: d_i = sum_j A_ij."""
    return np.diag(affinity.sum(axis=1))


def labels(clustering_matrix):
    """Cluster labels, c_i = argmax_j H[j,i], taken from the columns of H."""
    return np.argmax(clustering_matrix, axis=0)


def accuracy(labels_true, labels_predicted):
    """
    Clustering accuracy under the optimal assignment of cluster indices to
    ground-truth labels.

    Cluster indices are arbitrary, so the assignment that maximises agreement is
    found first, with the Hungarian algorithm.

    Returns a percentage.
    """
    from scipy.optimize import linear_sum_assignment

    labels_true = np.asarray(labels_true).ravel()
    labels_predicted = np.asarray(labels_predicted).ravel()
    n_points = labels_true.size

    unique_true = np.unique(labels_true)
    unique_predicted = np.unique(labels_predicted)
    size = max(unique_true.size, unique_predicted.size)

    # Contingency table: rows are predicted clusters, columns ground-truth ones.
    contingency = np.zeros((size, size))
    row_of_predicted = {label: i for i, label in enumerate(unique_predicted)}
    column_of_true = {label: i for i, label in enumerate(unique_true)}
    for predicted, true in zip(labels_predicted, labels_true):
        contingency[row_of_predicted[predicted], column_of_true[true]] += 1

    # Maximise agreement, so minimise its negative.
    rows, columns = linear_sum_assignment(-contingency)
    return 100.0 * contingency[rows, columns].sum() / n_points
