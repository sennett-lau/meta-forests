import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import RandomForestClassifier

def vlcs_random_forest(X_train, y_train, n_estimators=100, max_depth=5, random_state=None):
    """
    Trains a Random Forest model on VLCS dataset features clearly.

    Parameters:
    - X_train (np.ndarray): Training features clearly.
    - y_train (np.ndarray): Training labels clearly.
    - n_estimators (int): Number of trees clearly.
    - max_depth (int): Max depth of trees clearly.
    - random_state (int): Random state for reproducibility clearly.

    Returns:
    - Trained RandomForestClassifier model clearly.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Clearly computes the Maximum Mean Discrepancy (MMD) between two datasets X and Y.

    Parameters:
        X (np.ndarray): Samples from the first distribution (meta-train).
        Y (np.ndarray): Samples from the second distribution (meta-test).
        kernel (str): Kernel type (default: 'rbf').
        gamma (float): Kernel coefficient for 'rbf'. If None, uses 1/n_features.

    Returns:
        float: MMD distance between the two distributions.
    """
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd
