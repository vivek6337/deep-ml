import numpy as np

def sigmoid(x):
	return 1/(1+(np.exp(-1*x)))
def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	bias = np.array([bias]).reshape(-1 , 1)
	weights = weights.reshape(-1 , 1)
	z = X@weights + bias
	y_hat = sigmoid(z).reshape(-1)
	y_hat = np.array([1 if i>=0.5 else 0 for i in y_hat])

	
	return y_hat