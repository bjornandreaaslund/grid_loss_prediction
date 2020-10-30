import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# we evaluate on the same metrics used in the paper
# MAE, RMSE and MAPE
def evaluate(y_observed, y_true, y_pred) -> (float, float, float):
    """
    y_true: array-like of shape
    y_pred: array-like of shape
    return: mae, rmse, mape
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

<<<<<<< HEAD
=======
    # create dataframe that contains all values, where gaps are filled with NaN
    # NaN values are not plotted by seaborn, which allows us to combine all three 
    # timeseries in one plot
    nan_1 = np.empty(len(y_observed))
    nan_1[:] = np.nan
    nan_2 = np.empty(len(y_pred))
    nan_2[:] = np.nan

    observed_val = np.append(y_observed, nan_2)
    true_val = np.append(nan_1, y_true)
    pred_val = np.append(nan_1, y_pred)

    data = pd.dataframe({
        'y_observed': observed_val,
        'y_true': true_val,
        'y_pred': pred_val
    })

    # plot prediction and actual values
    sb.set()
    sb.lineplot(data=data)
    plt.show()
    
>>>>>>> 2f9aaa15dd2095a947ff9732e56de5dad6cc468a
    return mae, rmse, mape
