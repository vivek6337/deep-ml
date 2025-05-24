import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    t = matrix[0][0] + matrix[1][1]
    d = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    a1 = (t + np.sqrt(t**2 - 4*d))/2
    a2 = (t - np.sqrt(t**2 - 4*d))/2
    eigenvalues = [a1 , a2]
	return eigenvalues