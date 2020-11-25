from typing import Any, List, Tuple, Union
import numpy as np
from numpy.lib.format import header_data_from_array_1_0
from numpy.lib.npyio import save
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.base.model import Results


# we evaluate on the same metrics used in the paper
# MAE, RMSE and MAPE
def evaluate(y_observed, y_true, y_pred, plot=True) -> Tuple[float, float, float, float]:
    '''
    y_true: array-like of shape
    y_pred: array-like of shape
    return: mae, rmse, mape
    '''
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    # create dataframe that contains all values, where gaps are filled with NaN
    # NaN values are not plotted by seaborn, which allows us to combine all
    # timeseries in one plot
    nan_1 = np.empty(len(y_observed))
    nan_1[:] = np.nan
    nan_2 = np.empty(len(y_pred))
    nan_2[:] = np.nan

    observed = np.append(y_observed, y_true)
    predicted = np.append(nan_1, y_pred)

    data = pd.DataFrame({
        'observed': observed,
        'predicted': predicted
    })
    save_path = Path('Log/eval_data')
    data.to_pickle(str(save_path))

    # plot prediction and actual values
    if plot:
        sb.set()
        sb.lineplot(data=data, dashes=False, palette='colorblind')
        plt.show()

    return mae, rmse, mape, smape


def evaluate_all(y_true: pd.DataFrame, y_pred: pd.DataFrame, save_dir: Path) -> None:
    '''
    Evaluates predictions and saves the result as csv file and latex table

    y_true:     true values from measurement
    y_pred      values predicted by a model
    save_dir:   path to directory that evaluation will be saved in

    '''
    grid_names = ['grid1-loss', 'grid2-loss', 'grid3-loss']
    number_of_predictions = len(y_pred)

    if not save_dir.exists():
        save_dir.mkdir()

    accumulative = {
        'y_true': np.zeros(number_of_predictions),
        'y_pred': np.zeros(number_of_predictions),
    }
    results: List[List[Any]] = []

    for grid in grid_names:
        accumulative['y_true'] += y_true[grid]
        accumulative['y_pred'] += y_pred[grid]

        mae = mean_absolute_error(y_true[grid], y_pred[grid])
        rmse = mean_squared_error(y_true[grid], y_pred[grid])
        mape = mean_absolute_percentage_error(y_true[grid], y_pred[grid])
        smape = symmetric_mean_absolute_percentage_error(y_true[grid], y_pred[grid])

        results.append([grid.split('-')[0], mae, rmse, mape, smape])

    mae = mean_absolute_error(accumulative['y_true'], accumulative['y_pred'])
    rmse = mean_squared_error(accumulative['y_true'], accumulative['y_pred'])
    mape = mean_absolute_percentage_error(accumulative['y_true'], accumulative['y_pred'])
    smape = symmetric_mean_absolute_percentage_error(accumulative['y_true'], accumulative['y_pred'])

    results.append(['Accumulative', mae, rmse, mape, smape])

    df = pd.DataFrame(results, columns=['Grid_name', 'MAE', 'RMSE', 'MAPE', 'sMAPE']).set_index('Grid_name')
    df.to_csv(save_dir.joinpath('evaluation.csv'))
    df.to_latex(save_dir.joinpath('evaluation.tex'), caption='Evaluation of INSERT_MODEL')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))*200
