'''
Contents:

- Imports
- General settings
- Read data
- Transformation of data
- Training
- Evaluation

'''

from pathlib import Path
from nbeats_keras.model import NBeatsNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as joblib
from pathlib import Path
from evaluation_metric import evaluate
from tqdm import tqdm
from evaluation import evaluate, mean_absolute_percentage_error
savedir_models = Path('Models/').resolve()
loaddir = Path('Data/').resolve()
import matplotlib.pyplot as plt


def create_data(time_series, forecast_length, backcast_length, point = True):
    x = []
    y = []
    examples = len(time_series)-(backcast_length+forecast_length)
    if point == False:
        for i in range(examples):
            x.append(np.array(time_series[i:i+backcast_length]))
            y.append(np.array(time_series[i+backcast_length:i+backcast_length+forecast_length]))
        x = np.array(x).reshape(examples,backcast_length, 1)
        y = np.array(y).reshape(examples, forecast_length, 1)
    if point == True:
        for i in range(examples):
            x.append(time_series[i:i+backcast_length])
            y.append([i+backcast_length+forecast_length])
        x = np.array(x).reshape(examples,backcast_length, 1)
        y = np.array(y).reshape(examples, 1, 1)
    return x, y


def main():

    # https://keras.io/layers/recurrent/
    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.


    ''' Read data '''

    observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
    test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)
    y_true = test_true[['grid1-loss']]
    y_observed = observed[['grid1-loss']][10000:]

    scaler = joblib.load(savedir_models / 'scaler_grid1.sav')

    pickle_path1 = Path('Data/serialized/x_train').resolve()
    X_train = pd.read_pickle(pickle_path1)
    x = np.array(X_train['grid1-loss'].head(-6*24))
    exo = np.array(X_train['grid1-temp'].head(-6*24))

    pickle_path2 = Path('Data/serialized/y_train_grid1').resolve()
    y_train = pd.read_pickle(pickle_path2)
    y = np.array(y_train.head(-6*24))

    num_samples = X_train.shape[0]-6*24
    backcast_length, input_dim, output_dim = 6*24, 1, 6*24

    x = create_data(x, output_dim, backcast_length, False)
    exo = create_data(exo, output_dim, backcast_length, False)
    y = create_data(y, output_dim, backcast_length, False)


    ''' Create and train the model '''

    # Definition of the model.
    model = NBeatsNet(backcast_length=backcast_length, forecast_length=output_dim, exo_dim=1,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                      thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model.compile_model(loss='mae', learning_rate=1e-5)


    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, exo_train, y_train, x_val, exo_val, y_val = x[c:], exo[c:], y[c:], x[:c], exo[:c], y[:c]
    print(x_train)
    print(exo_train)
    print(exo_val)

    # Train the model.
    model.fit([x_train, exo_train], y_train, validation_data=([x_val, exo_val], y_val), epochs=20, batch_size=128)

    # Save the model for later.
    model.save('n_beats_model.h5')


    ''' Make predictions '''

    # Predict on the testing set.
    pickle_path = Path('Data/serialized/x_test').resolve()
    test = pd.read_pickle(pickle_path)
    X_test = pd.concat([X_train['grid1-loss'].tail(18*24), test['grid1-loss'].head(-18*24)])
    EXO_test = pd.concat([X_train['grid1-temp'].tail(18*24), test['grid1-temp'].head(-18*24)])

    predictions = model.predict([np.array(X_test).reshape(4369, 1, 1), np.array(EXO_test).reshape(4369, 1, 1)])
    predictions = predictions[:,0, 0]
    y_pred = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


    ''' Evaluate predictions '''

    mae, rmse, mape, smape = evaluate(np.array(y_observed), np.array(y_true), y_pred)
    # Load the model.

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("sMAPE", smape)


def demo():
    # https://keras.io/layers/recurrent/
    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    ''' General settings '''

    savedir_models = Path('Models/').resolve()
    loaddir = Path('Data/').resolve()
    nobs = 144 # 6 days
    lookback_window = 24*90 # 90 days
    grid_number = 0 # select which grid to predict : 0 -> grid1, 1 -> grid2, 2 -> grid3
    columns_to_predict = ['grid1-loss', 'grid2-loss', 'grid3-loss']
    target_col = columns_to_predict[grid_number] # select which column to predict

    ''' Read data '''

    # x-values
    pickle_path = Path('Data/serialized/x_train_with_lag').resolve()
    train = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/x_test_with_lag').resolve()
    test = pd.read_pickle(pickle_path)
    X_train = train['grid1-loss']
    X_test = test['grid1-loss']

    #y-values
    pickle_path = Path('Data/serialized/y_train_grid1').resolve()
    y_train_grid1 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid1').resolve()
    y_test_grid1 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_train_grid2').resolve()
    y_train_grid2 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid2').resolve()
    y_test_grid2 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_train_grid3').resolve()
    y_train_grid3 = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/y_test_grid3').resolve()
    y_test_grid3 = pd.read_pickle(pickle_path)

    # creates dataframes for evaluation
    observed = pd.read_csv(loaddir.joinpath('raw/train.csv'), header=0)
    test_true = pd.read_csv(loaddir.joinpath('raw/test.csv'), header=0)
    y_true = test_true[columns_to_predict]
    y_observed = observed[columns_to_predict][10000:]

    # scalers for inverse transform after predicting
    scaler_grid1 = joblib.load(savedir_models / 'scaler_grid1.sav')
    scaler_grid2 = joblib.load(savedir_models / 'scaler_grid2.sav')
    scaler_grid3 = joblib.load(savedir_models / 'scaler_grid3.sav')

    y_train = [y_train_grid1, y_train_grid2, y_train_grid3][grid_number] # select the y_train for the selected grid
    y_test = [y_test_grid1, y_test_grid2, y_test_grid3][grid_number]
    scaler = [scaler_grid1, scaler_grid2, scaler_grid3][grid_number] # select the scaler for the selected grid

    num_samples = 10754
    time_steps, input_dim, output_dim = 1, 1, 1

    def create_data(time_series, forecast_length, backcast_length, point = True):
        x = []
        y = []
        examples = len(time_series)-(backcast_length+forecast_length)
        if point == False:
            for i in range(examples):
                x.append(time_series[i:i+backcast_length])
                y.append(time_series[i+backcast_length:i+backcast_length+forecast_length])
            x = np.array(x).reshape(examples,backcast_length, 1)
            y = np.array(y).reshape(examples, forecast_length, 1)
        if point == True:
            for i in range(examples):
                x.append(time_series[i:i+backcast_length])
                y.append([i+backcast_length+forecast_length])
            x = np.array(x).reshape(examples,backcast_length, 1)
            y = np.array(y).reshape(examples, 1, 1)
        return x, y

    def create_test_data(time_series, backcast_length):
        x = []
        examples = len(time_series)-(backcast_length)
        for i in range(examples):
            x.append(time_series[i:i+backcast_length])
        x = np.array(x).reshape(examples,backcast_length, 1)
        return x
    forecast_length, backcast_length = 1, 24
    x, y = create_data(np.array(y_train_grid1), 1, 24, point=False)
    print(x.shape)
    print(y.shape)

    # Split data into training and testing datasets.
    c = x.shape[0] // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]

    # Definition of the model.
    #model = NBeatsNet(backcast_length=backcast_length, forecast_length=forecast_length,
    #                  stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
    #                  thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    #model.compile_model(loss='mae', learning_rate=1e-5)

    # Train the model.
    #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

    #__________________________________________
    y_train = [y_train_grid1, y_train_grid2, y_train_grid3][grid_number] # select the y_train for the selected grid
    lookback_window = 24*90 # 90 days
    y_test = [y_test_grid1, y_test_grid2, y_test_grid3][grid_number]
    exo = pd.concat([train, test])[['grid1-load']].tail(X_test.shape[0]+lookback_window)
    timeseries = pd.concat([y_train, y_test]).tail(X_test.shape[0]+lookback_window)

    forecast_length = 6*24
    backcast_length = 6*24
    output_dim = 1

    exo_test_x = create_test_data(exo.tail(X_test.shape[0]+backcast_length), backcast_length)
    test_x = create_test_data(timeseries.tail(X_test.shape[0]+backcast_length), backcast_length)

    pred = []

    print(exo_test_x.shape)
    print(test_x.shape)
    print(len(X_test))


    print(len(timeseries))

    print(len(timeseries)-24*90)


    for i in tqdm(range(len(timeseries)-lookback_window)):
        # fit new model on the sliding window
        # the model does not handle conastant values so we can not use binary variable
        # if they are constant in the forecast window
        # important to have the sesonal time series, but check that they are not constant
        forecast_slice = np.array(timeseries.iloc[i:lookback_window+backcast_length+i-1]).reshape(ookback_window+backcast_length-1, 1, 1)
        exo_slice = np.array(exo.iloc[i:lookback_window+backcast_length+i-1]).reshape(ookback_window+backcast_length-1, 1, 1)
        x, y = create_data(forecast_slice, forecast_length, backcast_length, point=True)
        exo_x, exo_y = create_data(exo_slice, forecast_length, backcast_length, point=True) # do not need exo_y

        if (i%24 == 0):
            model = NBeatsNet(backcast_length=backcast_length, exo_dim=1, forecast_length=output_dim,
                              stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                              thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)
            model.compile_model(loss='mae', learning_rate=1e-5)

            # trains one model each day
            model.fit([x, exo_x], y, epochs=10, batch_size=128)


        arr1 = np.array([test_x[i][:][:]])
        arr2 = np.array([exo_test_x[i][:][:]])
        pred_value = model.predict([arr1, arr2])
        print(pred_value[:,0, 0])
        pred.append(pred_value[:,0, 0])

    ''' Evaluate '''
    y_pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
    mae, rmse, mape = evaluate(np.array(y_observed[target_col]), np.array(y_true[target_col]), np.array(y_pred))
    print("\nEvaluate...")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)

from nbeats_keras.model import NBeatsNet

if __name__ == '__main__':
    main()
