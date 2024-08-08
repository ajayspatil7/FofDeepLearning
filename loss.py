import numpy as np


def l1_Loss(y, y_pred) -> float:
    return np.sum(abs(y - y_pred))

y = np.array([1,1,1,1,1])
y_pred = np.array([1,1,1,1,1])
l1_Loss(y, y_pred)



def l2_loss(y, y_pred) -> float:
    return np.sum(abs(y - y_pred)**2)


l2_loss(np.array([1, 0, 0, 1, 1]), np.array([0.9, 0.2, 0.1, 0.4, 0.9]))

