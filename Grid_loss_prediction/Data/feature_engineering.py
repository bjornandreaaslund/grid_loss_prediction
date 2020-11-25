import pandas as pd


def create_lag_features(lags, lag_variables, data):
    # we will keep the results in this dataframe
    data = pd.concat([x_train, x_test])
    data_with_lag = data.copy()
    for lag in lags:
        for var in lag_variables:
            data_with_lag[var+str(lag)] = data_with_lag[var].shift(lag*24)
    data_with_lag = data_with_lag.dropna()
    return data_with_lag
