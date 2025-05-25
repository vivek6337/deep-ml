import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
    w = np.array(initial_weights).reshape(-1,1)
    x = np.array(features)
    b = initial_bias
    mse_values = []
    for i in range(epochs):
        z = x @ w + b
        y_hat = 1/(1+np.exp(-z))
        y = np.array(labels).reshape(-1,1)
        loss = np.sum((y_hat - y)**2)/(len(labels))
        mse_values.append(loss)
        d_w = (x.T @ (2*y_hat*(y_hat-y)*(1-y_hat)))/len(labels)
        d_b = np.sum(2*y_hat*(y_hat-y)*(1-y_hat))/len(labels)
        w = w - learning_rate*d_w
        b = b - learning_rate*d_b
    
    updated_weights = np.round(w.flatten(),4).tolist()
    updated_bias = np.round(b,4)
    mse_values = np.round(mse_values,4).tolist()
          
	return updated_weights, updated_bias, mse_values