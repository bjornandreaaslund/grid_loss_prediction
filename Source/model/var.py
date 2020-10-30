import numpy as np
import pandas as pd
import joblib as joblib
from pathlib import Path
from statsmodels.tsa.api import VAR
from evaluation_metric import evaluate

savedir_models = Path('Models/').resolve()
loaddir = Path('Data/').resolve()
scaler_filename = "scaler_test.save"
nobs = 24
lookback_window = 336
columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']

# %% ------------------------------- Read data ------------------------------- #

pickle_path = Path('Data/serialized/processed_x_train_scaled_pickle').resolve()
train = pd.read_pickle(pickle_path)
train = train.dropna()

pickle_path = Path('Data/serialized/processed_x_test_scaled_pickle').resolve()
test = pd.read_pickle(pickle_path)

observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)

# %% ------------------------------- Vector Autoregressive ------------------------------- #
var = VAR(np.array(train))
var_fitted = var.fit()
print(var_fitted.summary())

# %% ------------------------------- Forecast ------------------------------- #
frames = [train.tail(lookback_window), test]
df_test = pd.concat(frames)

# scaling back to original scaling
scaler_test = joblib.load(savedir_models / scaler_filename)

pred = pd.DataFrame(columns = columns_to_predict)

for i in range(test.shape[0]):
    forecast_input = df_test.values[i:lookback_window+i]
    fc = var_fitted.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=train.index[-nobs:], columns=train.columns)
    df_forecast[df_forecast.columns] = scaler_test.inverse_transform(df_forecast)
    pred_values = df_forecast.tail(1)
    pred = pred.append(pred_values[columns_to_predict], ignore_index=True)

print(pred)


# %% ------------------------------- Evaluate ------------------------------- #

for col in pred.columns:
    y_pred = np.array(pred[col])
    y_true = np.array(test_true[col])
    y_observed = np.array(observed[col][10000:])
    mae, rmse, mape = evaluate(y_observed, y_true, y_pred)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
