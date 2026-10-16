import numpy as np

class Node:
    def __init__(self , feature_idx=None , thresh=None , leaf_value=None):
        self.feature_idx = feature_idx
        self.thresh = thresh
        self.left = None
        self.right = None
        self.leaf_value = leaf_value


def gini_impurity(y):
	g = 1
	n = y.shape[0]
	d = {}
	for i in y:
		if i in d:
			d[i]+=1
		else:
			d[i] = 1

	for k , v in d.items():
		g -= (v/n)**2
	return g

def grow(x , y , f):
    split_cost = np.inf
    best_thresh = 0
    best_feat = 0
    best_idx = 0
    for i in f:
        idx = np.argsort(x[: , i])
        x_f = x[idx , i]
        y_f = y[idx]
        n = x_f.shape[0]
        for j in range(n-1):
            s = ((j+1)/n)*gini_impurity(y_f[:j+1]) + ((n-(j+1))/n)*gini_impurity(y_f[j+1:])
            if s < split_cost:
                split_cost = s
                best_feat = i
                best_thresh = (x_f[j]+x_f[j+1])/2
                best_idx = j

    return best_feat , best_thresh , best_idx

def split(x , y , f , thresh , best_idx):
    idx = np.argsort(x[: , f])
    x = x[idx]
    y = y[idx]
    return x[:best_idx+1] , y[:best_idx+1] , x[best_idx+1 :] , y[best_idx+1 :]


def build_tree(x , y , depth , max_depth , min_samples_split , max_features , rng ):
	if((x.shape[0]<min_samples_split) or (depth==max_depth) or (gini_impurity(y)==0)):
		d = {}
		for i in y:
			if i in d:
				d[i] += 1
			else:
				d[i] = 1
		return Node(leaf_value = min(d , key = lambda k : (-d[k] , k)))
	f = rng.choice(x.shape[1] , size = min(x.shape[1] , max_features) , replace = False)
	b_f , b_t , b_i = grow(x , y , f)
	x1 , y1 , x2 , y2 = split(x , y , b_f , b_t , b_i)
	n = Node(b_f , b_t)
	n.left = build_tree(x1 , y1 , depth+1 , max_depth , min_samples_split , max_features , rng )
	n.right = build_tree(x2 , y2 , depth+1 , max_depth , min_samples_split, max_features , rng )
	return n


def predict(node , x):
    if(node.left==None and node.right==None):
        return node.leaf_value
    f = node.feature_idx
    t = node.thresh
    if(x[f]<=t):
        return predict(node.left , x)
    else:
        return predict(node.right , x)

def boot_agg(x , y , rng):
	n = x.shape[0]
	m = x.shape[1]
	row_idx = rng.choice(n , size=n , replace = True)
	x = x[row_idx]
	y = y[row_idx]
	return x , y


def random_forest_fit_predict(X_train, y_train, X_test, n_trees: int = 5, max_depth: int = 3, max_features: int = 2, seed: int = 0):
	"""
	Train a random forest classifier and predict on X_test.
	Returns a list of integer class predictions.
	"""
	# Your code here
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	rng = np.random.RandomState(seed = seed)
	tree_l = []
	for i in range(n_trees):
		x , y = boot_agg(X_train , y_train , rng)
		root = build_tree(x , y , 0 , max_depth , 2 , max_features, rng)
		tree_l.append(root)
	
	predictions = []
	for i in range(X_test.shape[0]):
		p = {}
		for tree in tree_l:
			c = predict(tree , X_test[i])
			if c in p:
				p[c] += 1
			else:
				p[c] = 1
		predictions.append((int)(min(p , key=lambda k : (-p[k] , k))))

	return predictions
