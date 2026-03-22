"""
data_utils.py
─────────────────────────────────────────────────────────────
Handles synthetic dataset generation for the ML comparison app.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_dataset(
    n_samples: int = 300,
    class_sep: float = 1.0,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """
    Generate a 2-class, 2-feature synthetic classification dataset.

    Parameters
    ----------
    n_samples    : Total number of data points.
    class_sep    : Higher values → easier to separate classes.
    random_state : Seed for reproducibility.
    test_size    : Fraction of data reserved for testing.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test  – split arrays
        X_all, y_all                       – full (unscaled) for plotting
        scaler                             – fitted StandardScaler
        feature_range                      – (x_min, x_max, y_min, y_max) for plots
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=random_state,
    )

    # Labels must be –1 / +1 for the custom Perceptron
    y_signed = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_signed, test_size=test_size, random_state=random_state, stratify=y_signed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    margin = 0.5
    feature_range = (
        X_all_s[:, 0].min() - margin,
        X_all_s[:, 0].max() + margin,
        X_all_s[:, 1].min() - margin,
        X_all_s[:, 1].max() + margin,
    )

    return {
        "X_train": X_train_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "X_all": X_all_s,
        "y_all": y_signed,
        "scaler": scaler,
        "feature_range": feature_range,
    }