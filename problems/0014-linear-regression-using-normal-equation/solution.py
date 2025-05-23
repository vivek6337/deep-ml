import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    # X = np.hpstack(np.ones(X.shape[0],1),X) already there
    X = np.array(X)
    y = np.array(y)
    theta = np.linalg.inv(X.T @ X) @ (X.T) @ y
    theta = np.round(theta)
	return theta