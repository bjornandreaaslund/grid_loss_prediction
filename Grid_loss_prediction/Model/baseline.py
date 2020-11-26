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

    LOADDIR = Path('Data/').resolve()
    NOBS = 144 # 6 days
    COLUMNS_TO_PREDICT = ['grid1-loss', 'grid2-loss', 'grid3-loss']


    '''Read data'''

    train = pd.read_csv(LOADDIR.joinpath('raw/train.csv'), header=0)
    test = pd.read_csv(LOADDIR.joinpath('raw/test.csv'), header=0)

    frames = [train.tail(24+NOBS), test] # we will forecast 6 days ahead, and fit the model on all available data up to this point
    df_test = pd.concat(frames)
    y_pred = df_test[COLUMNS_TO_PREDICT].head(test.shape[0])

    '''Save predictions'''

    np.savetxt('Data/predictions/baseline/y_pred_baseline.csv', y_pred, delimiter = ',')


if __name__ == "__main__":
    main()
