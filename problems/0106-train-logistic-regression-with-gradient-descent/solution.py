import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	m , n = X.shape
	y = y.reshape(-1 , 1)
	w = np.zeros((n , 1))
	b = np.zeros((1 , 1))
	eps = 1e-8
	lr = learning_rate
	loss_l = []
	for i in range(iterations):
		z = X@w + b
		y_hat = sigmoid(z)
		loss = -1*np.sum(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))
		loss_l.append(round(loss , 4))
		dl_dz = (y_hat - y)
		dl_dw = np.transpose(X)@dl_dz
		dl_db = np.sum(dl_dz , axis = 0)
		w = w - lr*dl_dw
		b = b - lr*dl_db
	w = np.round(w.reshape(-1) , 4)
	b = np.round(b.reshape(-1) , 4)
	w = w.tolist()
	b = b.tolist()
	b.extend(w)
	params = b

	return (params , loss_l)