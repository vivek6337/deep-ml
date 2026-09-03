import numpy as np

def soft_shrinkage(w , thresh):
    for i in range(w.shape[0]):
        s = -1 if w[i][0]<0 else 1
        w[i][0] = s*max(abs(w[i][0])-thresh , 0)
    return w

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    m , n = X.shape
    y = y.reshape(-1 , 1)
    w = np.zeros((n,1))
    b = np.zeros((1 , 1))
    lr = learning_rate

    for i in range(max_iter):
        y_hat = X@w + b
        loss = np.mean((y_hat - y)**2)/2 + alpha2*np.linalg.norm(w)**2 + alpha1*np.linalg.norm(w , ord = 1)
        dl_dy = (y_hat - y)/m
        dl_dw = np.transpose(X)@dl_dy + 2*alpha2*w + alpha1*np.sign(w)
        dl_db = np.sum(dl_dy , axis = 0)
        b = b - lr*dl_db
        w_temp = w - lr*dl_dw
        w_new = w_temp
        # w_new = soft_shrinkage(w_temp , lr*alpha1)
        if(np.linalg.norm(dl_dw , ord = 1) < tol):
            w = w_new
            break
        w = w_new


    return (w.squeeze(1), b[0][0])