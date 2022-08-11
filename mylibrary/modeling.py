from msilib.schema import Error
from sklearn.metrics import mean_squared_error
import numpy as np


######################### python Error classes ##############################
class meanError(Exception):
    def __init__(self, msg='meanError: mean(y_true) is not Zero'):
        self.msg = msg
    
    def __str__(self):
        return self.msg
#############################################################################



################################ Functions ##################################
def RMSE(y_true, y_pred):
    """Calculate the Root Mean Squared Error."""
    return mean_squared_error(y_true, y_pred) ** 0.5

        
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score
#############################################################################