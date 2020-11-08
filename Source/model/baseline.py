'''
Contents:

- Imports
- General settings
- Read data
- Evaluation

'''

# %% --------------------------------- Imports --------------------------------- #

import numpy as np
import pandas as pd
import joblib as joblib
from pathlib import Path
from statsmodels.tsa.api import VAR
from evaluation_metric import evaluate
from tqdm import tqdm

# %% --------------------------------- General settings --------------------------------- #

savedir_models = Path('Models/').resolve()
loaddir = Path('Data/').resolve()
nobs = 144 # 6 days
lookback_window = 7320 # 180 days
columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']

# %% ------------------------------- Read data ------------------------------- #

observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)

# %% ------------------------------- Evaluate ------------------------------- #

# we will forecast 6 days ahead, and fit the model on all available data up to this point
frames = [observed.tail(24+nobs), test_true]
df_test = pd.concat(frames)

pred = []

for col in columns_to_predict:
    y_pred = np.array(df_test[col].head(test_true.shape[0]))
    y_true = np.array(test_true[col])
    y_observed = np.array(observed[col][10000:])
    mae, rmse, mape = evaluate(y_observed, y_true, y_pred)
    print(col)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
