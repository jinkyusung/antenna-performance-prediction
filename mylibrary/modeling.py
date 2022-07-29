from sklearn.metrics import mean_squared_error
import numpy as np


def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def NRMSE(y_true, y_pred, divisor='max-min', percentage=True):
    """Calculate the Normalized Root Mean Squared Error."""
    
    result = RMSE(y_true, y_pred)
    if divisor == 'max-min':
        max_y_true = max(y_true)
        min_y_true = min(y_true)
        if max_y_true <= min_y_true:
            raise "ValueError: divisor is not Zero."
        else:
            result /= max_y_true - min_y_true

    elif divisor == 'mean':
        result /= np.mean(y_true)

    else:
        raise "divisor should be 'max-min' or 'mean'."

    if percentage:
        result *= 100
    return result

