'''
Contents:

- Imports
- General settings
- Read data
- Vector Autoregressive
- Forecasting
- Evaluation

'''

# %% --------------------------------- Imports --------------------------------- #

import numpy as np
import pandas as pd
import joblib as joblib
from pathlib import Path
from statsmodels.tsa.api import VAR
from evaluation_metric import evaluate, evaluate_all
from tqdm import tqdm


savedir_models = Path('Models/').resolve()
loaddir = Path('Data/').resolve()
scaler_filename = "scaler_test.sav"
nobs = 144 # 6 days
lookback_window = 7320 # 180 days
columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']

# %% ------------------------------- Read data ------------------------------- #

pickle_path = Path('Data/serialized/x_train').resolve()
train = pd.read_pickle(pickle_path)
train = train.dropna()

pickle_path = Path('Data/serialized/x_test').resolve()
test = pd.read_pickle(pickle_path)

observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)

# %% --------------------------- Vector Autoregressive ----------------------- #
# need to find the best order p by hyperparameter optimization

var = VAR(np.array(train))

res = var.select_order(maxlags=10)
print(res.summary())

# from the output above we can see which order the model chooses
# the model chooses the order with minimum information criteria
var_fitted = var.fit(maxlags=10)
#print(var_fitted.summary())

# %% ------------------------------- Forecast ------------------------------- #
# we will forecast 6 days ahead, and fit the model on all available data up to this point
frames = [train.tail(lookback_window+nobs), test]
df_test = pd.concat(frames)

# scaling back to original scaling
scaler_test = joblib.load(savedir_models / scaler_filename)

pred = pd.DataFrame(columns = columns_to_predict)

for i in tqdm(range(test.shape[0])):
    # fit new model on the sliding window
    # the model does not handle conastant values so we can not use binary variable
    # if they are constant in the forecast window
    # important to have the sesonal time series, but check that they are not constant
    forecast_df = df_test.iloc[i:lookback_window+i]
    forecast_input = np.array(forecast_df)
    var = VAR(forecast_input)
    var_fitted = var.fit(maxlags=10)
    fc = var_fitted.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=train.index[-nobs:], columns=train.columns)
    df_forecast[df_forecast.columns] = scaler_test.inverse_transform(df_forecast) # scale back after forecasting
    pred_values = df_forecast.tail(1)
    pred = pred.append(pred_values[columns_to_predict], ignore_index=True)

#print(pred)


# %% ------------------------------- Evaluate ------------------------------- #


y_true = test_true[pred.columns]
y_pred = pred

evaluate_all(y_true, y_pred)

# for col in pred.columns:
#     y_pred = np.array(pred[col])
#     y_true = np.array(test_true[col])
#     y_observed = np.array(observed[col][10000:])
#     mae, rmse, mape, smape = evaluate(y_observed, y_true, y_pred)
#     print(col)
#     print("MAE:", mae)
#     print("RMSE:", rmse)
#     print("MAPE:", mape)
#     print("SMAPE", smape)

# # Evaluate sum
# y_pred = np.array(pred.sum(axis=1))
# y_true = np.array(test_true[columns_to_predict].sum(axis=1))
# y_observed = np.array(observed[columns_to_predict].sum(axis=1)[10000:])
# mae, rmse, mape, smape = evaluate(y_observed, y_true, y_pred)
# print("Total")
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("MAPE:", mape)
# print("SMAPE", smape)
