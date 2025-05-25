import numpy as np
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	# Your code here
    initial_centroids = np.array(initial_centroids)
    points = np.array(points)
    for i in range(max_iterations):
        i_c = np.repeat(initial_centroids[:,np.newaxis , :] , len(points) , axis = 1)
        p = np.repeat(points[np.newaxis , : , :] , k , axis = 0)
        n = np.linalg.norm(i_c-p,axis = 2)
        idx = np.argmin(n , axis = 0)
        b = np.bincount(idx)
        new_centroids = np.array([np.zeros(points.shape[1]) for _ in range(k)] )
        for j in range(len(points)):
            new_centroids[idx[j]] += points[j]/b[idx[j]]
        if np.allclose(new_centroids , initial_centroids):
             break
        else:
            initial_centroids = new_centroids
        
    new_centroids = new_centroids.tolist()
    final_centroids = [tuple(np.roun