from msilib.schema import Error
from sklearn.metrics import mean_squared_error
import numpy as np


######################### python Error classes ##############################
class maxminError(Exception):
    def __init__(self):
        self.msg = 'maxminError: max(y_true) is greater than min(y_true)'
    
    def __str__(self):
        return self.msg


class meanError(Exception):
    def __init__(self, msg='meanError: mean(y_true) is not Zero'):
        self.msg = msg
    
    def __str__(self):
        return self.msg


class keywordError(Exception):
    def __init__(self, user_input=""):
        self.msg = "keywordError: invalid keyword for NRMSE()"
        self.msg += f"""divisor should be 'max-min' or 'mean
                        your input is '{user_input}'."""
    
    def __str__(self):
        return self.msg
#############################################################################




################################ Functions ##################################
def RMSE(y_true, y_pred):
    """Calculate the Root Mean Squared Error."""
    return mean_squared_error(y_true, y_pred) ** 0.5


def NRMSE(y_true, y_pred, divisor='max-min', percentage=True):
    """Calculate the Normalized Root Mean Squared Error."""
    try:
        result = RMSE(y_true, y_pred)
        if divisor == 'max-min':
            max_y_true = max(y_true)
            min_y_true = min(y_true)
            if max_y_true <= min_y_true:
                raise maxminError()
            else:
                result /= max_y_true - min_y_true
        elif divisor == 'mean':
            result /= np.mean(y_true)
        else:
            raise keywordError(divisor)

        if percentage:
            result *= 100
        return result
    
    except maxminError:
        print(maxminError)
    
    except Exception:
        import traceback
        traceback.print_exc()
#############################################################################