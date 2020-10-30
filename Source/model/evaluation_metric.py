import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from from sklearn.metrics import mean_absolute_percentage_error

# we evaluate on the same metrics used in the paper
# MAE, RMSE and MAPE
def evaluate(y_observed, y_true, y_pred):
    """
    y_true: array-like of shape
    y_pred: array-like of shape
    return: mae, rmse, mape
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # TODO: create plot
    
    return mae, rmse, mape
