import numpy as np

class Node:
    def __init__(self , feature_idx=None , thresh=None , leaf_value=None):
        self.feature_idx = feature_idx
        self.thresh = thresh
        self.left = None
        self.right = None
        self.leaf_value = leaf_value


def mse(y):
    y_mean = np.mean(y)
    return np.mean((y-y_mean)**2)

def grow(x , y):
    split_cost = np.inf
    best_thresh = 0
    best_feat = 0
    best_idx = 0
    for i in range(x.shape[1]):
        idx = np.argsort(x[: , i])
        x_f = x[idx , i]
        y_f = y[idx]
        n = x_f.shape[0]
        for j in range(n-1):
            s = ((j+1)/n)*mse(y_f[:j+1]) + ((n-(j+1))/n)*mse(y_f[j+1:])
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


def build_tree(x , y , depth , max_depth , min_samples_split):
    if((x.shape[0]<min_samples_split) or (depth==max_depth)):
        return Node(leaf_value = np.mean(y))
    b_f , b_t , b_i = grow(x , y)
    x1 , y1 , x2 , y2 = split(x , y , b_f , b_t , b_i)
    n = Node(b_f , b_t)
    n.left = build_tree(x1 , y1 , depth+1 , max_depth , min_samples_split)
    n.right = build_tree(x2 , y2 , depth+1 , max_depth , min_samples_split)
    return n


def predict(node , x):
    if(node.left==None and node.right==None):
        return node.leaf_value
    f = node.feature_idx
    t = node.thresh
    if(x[f]<t):
        return predict(node.left , x)
    else:
        return predict(node.right , x)



def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    tree = build_tree(X_train , y_train , 0, max_depth , min_samples_split)
    y_pred = []
    for i in range(X_test.shape[0]):
        y_pred.append(np.round(predict(tree , X_test[i]) , 4))

    return y_pred
