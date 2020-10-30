import numpy as np
import pandas as pd
import joblib as joblib
from pathlib import Path
from statsmodels.tsa.api import VAR
from evaluation_metric import evaluate

savedir_models = Path('Models/').resolve()
scaler_filename = "scaler.save"
nobs = 10
lookback_window = 1000
columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']

# %% ------------------------------- Read data ------------------------------- #

pickle_path = Path('Data/serialized/processed_x_train_scaled_pickle').resolve()
train = pd.read_pickle(pickle_path)
train = train.dropna()

# %% ------------------------------- Vector Autoregressive ------------------------------- #
var = VAR(np.array(train))
var_fitted = var.fit()
print(var_fitted.summary())

# Forecast
forecast_input = train.values[-lookback_window:]
print(train.columns)

fc = var_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=train.index[-nobs:], columns=train.columns)
print(df_forecast)

# scaling back to original scaling
scaler = joblib.load(savedir_models / scaler_filename)
df_forecast[df_forecast.columns] = scaler.inverse_transform(df_forecast)
df_forecast = df_forecast[columns_to_predict]
print(df_forecast)
