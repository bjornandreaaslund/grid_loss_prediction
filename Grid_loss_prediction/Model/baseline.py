'''
This module creates a baseline model and forecasts the test data.
The predictions are saved in Data/predictions/baseline/.

Contents:

- Imports
- General settings
- Read data
- Save predictions

'''

import numpy as np
import pandas as pd
import joblib as joblib
from pathlib import Path
from statsmodels.tsa.api import VAR


def main():

    '''General settings'''

    savedir_models = Path('Models/').resolve()
    loaddir = Path('Data/').resolve()
    nobs = 144 # 6 days
    lookback_window = 7320 # 180 days
    columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']


    '''Read data'''

    train = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
    test = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)

    frames = [train.tail(24+nobs), test] # we will forecast 6 days ahead, and fit the model on all available data up to this point
    df_test = pd.concat(frames)
    y_pred = df_test[columns_to_predict].head(test.shape[0])

    '''Save predictions'''

    np.savetxt('Data/predictions/baseline/y_pred_baseline.csv', y_pred, delimiter = ',')


if __name__ == "__main__":
    main()
