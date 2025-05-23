import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))
    y = y.reshape(-1,1)
    for i in range(iterations):
        theta = theta - alpha * (X.T @ (X @ theta - y))/m
    theta = np.round(theta , 4)
	return theta