import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	if len(a)*len(a[0]) != new_shape[0]*new_shape[1]:
		return []

	else:
		l = []
		for i in range(len(a)):
			for j in range(len(a[0])):
				l.append(a[i][j])
		reshaped_matrix = []
		c = 0
		for i in range(new_shape[0]):
			f = []
			for j in range(new_shape[1]):
				f.append(l[c])
				c+=1
			reshaped_matrix.append(f)


	return reshaped_matrix