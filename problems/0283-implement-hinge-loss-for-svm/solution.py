import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average hinge loss for SVM classification.
    
    Args:
        y_true: Array of true labels (-1 or +1)
        y_pred: Array of predicted scores (raw SVM scores)
    
    Returns:
        Average hinge loss rounded to 4 decimal places
    """
    score = 0
    for i in range(y_true.shape[0]):
        score += max(0 , 1-(y_true[i]*y_pred[i]))

    return score/y_true.shape[0]