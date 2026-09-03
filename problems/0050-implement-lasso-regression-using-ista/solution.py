import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    for i in range(w.shape[0]):
        s = -1 if w[i] <0 else 1
        w[i] = s*max(abs(w[i])-threshold , 0)
    return w

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    m, n = X.shape
    w = np.zeros((n , 1))
    b = np.zeros((1,1))
    y = y.reshape(-1,1)
    lr = learning_rate
    
    # Your code here
    for i in range(max_iter):
        y_hat = X@w + b
        loss = np.mean((y_hat-y)**2)/2
        dl_dy = (y_hat - y)/m
        dl_dw = np.transpose(X)@dl_dy
        w_temp = w - lr*dl_dw
        w_new = soft_threshold(w_temp , lr*alpha)
        dl_db = np.sum(dl_dy , axis = 0)
        b = b - lr*dl_db
        if(np.linalg.norm(w_new - w) < tol):
            break
        w = w_new


    return (w.squeeze(1) , b.squeeze(1).item())
