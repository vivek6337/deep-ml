import numpy as np
from typing import Tuple

def gini_impurity(y):
    d = {}
    for i in range(y.shape[0]):
        if y[i] in d:
            d[y[i]]+=1
        else:
            d[y[i]] = 1
    g = 1
    n = y.shape[0]
    for k , v in d.items():
        g -= (v/n)**2
    return g

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    best_gain = np.inf
    best_idx = 0
    best_thresh = 0
    for i in range(X.shape[1]):
        idx = np.argsort(X[: , i])
        x_f = X[idx , i]
        y_f = y[idx]
        n = x_f.shape[0]
        for j in range(n-1):
            if j>0 and x_f[j-1]==x_f[j]:
                continue
            g = (((j+1)/n)*gini_impurity(y_f[:j+1]) + ((n-(j+1))/n)*gini_impurity(y_f[j+1:]))
            if g<best_gain:
                best_gain = g
                best_idx = i
                best_thresh = x_f[j]

    return (best_idx , best_thresh)




