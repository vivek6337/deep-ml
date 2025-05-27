import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	# Your code here
    mean = np.mean(data , axis= 0)
    std = np.std(data , axis = 0)
    std_data = (data-mean)/std
    co_m = np.dot(std_data.T , std_data)/data.shape[0]
    e_values , e_vectors = np.linalg.eig(co_m)
    sorted_idx = np.argsort(e_values)
    e_vectors = e_vectors[:,sorted_idx[::-1]]
    principal_components = e_vectors[:,:k]
	return np.round(principal_components, 4)