'''
This module creates a N-BEATS model and trains the model.
An ensemble of models is then used to create a prediction, and the predictions are saved in Data/predictions/nbeats/.

Contents:

- General settings
- Read data
- Transformation of data
- Training
- Save predictions

'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as joblib
from pathlib import Path
from tqdm import tqdm

from nbeats_keras.model import NBeatsNet

def main():

    ''' General settings '''

    SAVEDIR_MODELS = Path('Models/').resolve()
    LOADDIR = Path('Data/').resolve()
    FORECAST_LENGTH = 144 # 6 days
    LOOKBACK_WINDOWS = FORECAST_LENGTH*np.arange(1,5) # n*H where n is 1, 2, 3, 4
    INPUT_DIM=1 # the number of timeseries we want to forecast
    EXO_DIM=4 # the number of timeseries used to make a prediction, but will not be forecasted
    GRID_NUMBER = 2 # select which grid to predict : 0 -> grid1, 1 -> grid2, 2 -> grid3
    COLUMNS_TO_PREDICT = ['grid1-loss', 'grid2-loss', 'grid3-loss']
    exo1 = ['grid1-load', 'grid2-load', 'grid3-load'][GRID_NUMBER]
    exo2 = ['grid1-temp', 'grid2_1-temp', 'grid3-temp'][GRID_NUMBER]
    exo3 = 'holiday'
    exo4 = 'demand'
    TARGET_COL = COLUMNS_TO_PREDICT[GRID_NUMBER] # select which column to predict
    TRAINING = False

    ''' Read data '''

    # x-values
    pickle_path = Path('Data/serialized/x_train').resolve()
    X_train = pd.read_pickle(pickle_path)
    pickle_path = Path('Data/serialized/x_test').resolve()
    X_test = pd.read_pickle(pickle_path)

    # for evaluation
    observed = pd.read_csv(LOADDIR.joinpath('raw/train.csv'), header=0)
    test_true = pd.read_csv(LOADDIR.joinpath('raw/test.csv'), header=0)
    y_true = test_true[COLUMNS_TO_PREDICT]
    y_observed = observed[COLUMNS_TO_PREDICT][10000:]

    # scalers for inverse transform after predicting
    scaler_grid1 = joblib.load(SAVEDIR_MODELS / 'scaler_grid1.sav')
    scaler_grid2 = joblib.load(SAVEDIR_MODELS / 'scaler_grid2.sav')
    scaler_grid3 = joblib.load(SAVEDIR_MODELS / 'scaler_grid3.sav')

    scaler = [scaler_grid1, scaler_grid2, scaler_grid3][GRID_NUMBER] # select the scaler for the selected grid

    ''' Transformation of data '''

    def get_x_y_data(df, backcast_length, forecast_length):
        x = np.array([]).reshape(0, backcast_length)
        y = np.array([]).reshape(0, forecast_length)

        time_series = np.array(df[[TARGET_COL]])
        time_series = time_series.reshape(time_series.shape[0])

        time_series_cleaned_forlearning_x = np.zeros(
            (time_series.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        time_series_cleaned_forlearning_y = np.zeros(
            (time_series.shape[0] + 1 - (backcast_length + forecast_length), forecast_length))
        for j in range(backcast_length, time_series.shape[0] + 1 - forecast_length):
            time_series_cleaned_forlearning_x[j - backcast_length, :] = time_series[j - backcast_length:j]
            time_series_cleaned_forlearning_y[j - backcast_length, :] = time_series[j: j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

        return x.reshape((x.shape[0], x.shape[1], 1)), y.reshape((y.shape[0], y.shape[1], 1))

    def get_exo_var_data(df, backcast_length, forecast_length):
        e1 = np.array([]).reshape(0, backcast_length)
        e2 = np.array([]).reshape(0, backcast_length)
        e3 = np.array([]).reshape(0, backcast_length)
        e4 = np.array([]).reshape(0, backcast_length)

        time_series_1 = np.array(df[[exo1]])
        time_series_1 = time_series_1.reshape(len(time_series_1))
        time_series_2 = np.array(df[[exo2]])
        time_series_2 = time_series_2.reshape(len(time_series_2))
        time_series_3 = np.array(df[[exo3]])
        time_series_3 = time_series_3.reshape(len(time_series_3))
        time_series_4 = np.array(df[[exo4]])
        time_series_4 = time_series_4.reshape(len(time_series_4))

        time_series_cleaned_forlearning_1 = np.zeros(
            (time_series_1.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        time_series_cleaned_forlearning_2 = np.zeros(
            (time_series_2.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        time_series_cleaned_forlearning_3 = np.zeros(
            (time_series_3.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        time_series_cleaned_forlearning_4 = np.zeros(
            (time_series_4.shape[0] + 1 - (backcast_length + forecast_length), backcast_length))
        for j in range(backcast_length, time_series_1.shape[0] + 1 - forecast_length):
            time_series_cleaned_forlearning_1[j - backcast_length, :] = time_series_1[j - backcast_length:j]
            time_series_cleaned_forlearning_2[j - backcast_length, :] = time_series_2[j - backcast_length:j]
            time_series_cleaned_forlearning_3[j - backcast_length, :] = time_series_3[j - backcast_length:j]
            time_series_cleaned_forlearning_4[j - backcast_length, :] = time_series_4[j - backcast_length:j]
        e1 = np.vstack((e1, time_series_cleaned_forlearning_1))
        e2 = np.vstack((e2, time_series_cleaned_forlearning_2))
        e3 = np.vstack((e3, time_series_cleaned_forlearning_3))
        e4 = np.vstack((e4, time_series_cleaned_forlearning_4))

        return e1, e2, e3, e4

    def get_data(df, backcast_length, forecast_length, mode = "train"):
        x, y = get_x_y_data(df, backcast_length, forecast_length)
        e1, e2, e3, e4 = get_exo_var_data(df, backcast_length, forecast_length)

        e = np.concatenate((e1.reshape((e1.shape[0], e1.shape[1], 1)), e2.reshape((e2.shape[0], e2.shape[1], 1)), e3.reshape((e3.shape[0], e3.shape[1], 1)), e4.reshape((e4.shape[0], e4.shape[1], 1))), axis=-1)

        if mode == "train":
            return x[:90 * x.shape[0] // 100], e[:90 * x.shape[0] // 100], y[:90 * x.shape[0] // 100]
        elif mode == "test":
            return x[90 * x.shape[0] // 100:], e[90 * x.shape[0] // 100:], y[90 * x.shape[0] // 100:]
        elif mode == "pred":
            return x, e, y

    ''' Training '''

    def get_metrics(y_true, y_hat):
        error = np.mean(np.square(y_true - y_hat))
        smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
        return smape, error

    def ensure_results_dir():
        if not os.path.exists('Logs/results/test'):
            os.makedirs('Logs/results/test')

    def train_model(df, model: NBeatsNet, best_perf=np.inf, max_steps=10001, plot_results=100):
        ensure_results_dir()
        x_test, e_test, y_test = get_data(df, model.backcast_length, model.forecast_length, mode="test")

        for step in range(max_steps):
            x_train, e_train, y_train = get_data(df, model.backcast_length, model.forecast_length, mode="train")

            model.train_on_batch([x_train, e_train], y_train)

            if step % plot_results == 0:
                print('step=', step)
                model.save('Logs/results/n_beats_model_' + str(model.backcast_length) + str(step) + TARGET_COL + "_backcast_"  + '.h5')
                predictions = model.predict([x_train, e_train])
                validation_predictions = model.predict([x_test, e_test])

                smape = get_metrics(y_test, validation_predictions)[0]
                print('smape=', smape)
                if smape < best_perf:
                    best_perf = smape
                    model.save('Logs/results/n_beats_model_ongoing_' + str(model.backcast_length)+ TARGET_COL + "_backcast_" + '.h5')

                for k in range(model.input_dim):
                    plot_keras_model_predictions(model, False, step, x_train[0, :, k], y_train[0, :, k],predictions[0, :, k], axis=k)
                    plot_keras_model_predictions(model, True, step, x_test[0, :, k], y_test[0, :, k],validation_predictions[0, :, k], axis=k)

        model.save('Logs/results/n_beats_model_' + TARGET_COL + '.h5')

        predictions = model.predict([x_train, e_train])
        validation_predictions = model.predict([x_test, e_test])

        for k in range(model.input_dim):
            plot_keras_model_predictions(model, False, max_steps, x_train[10, :, k], y_train[10, :, k],predictions[10, :, k], axis=k)
            plot_keras_model_predictions(model, True, max_steps, x_test[10, :, k], y_test[10, :, k],validation_predictions[10, :, k], axis=k)

        print('smape=', get_metrics(y_test, validation_predictions)[0])
        print('error=', get_metrics(y_test, validation_predictions)[1])


    def plot_keras_model_predictions(model, is_test, step, backcast, forecast, prediction, axis):
        if is_test:
            title = 'Logs/results/test/' + TARGET_COL + "_backcast_" +str(len(backcast)) +  '_step_' + str(step) + '_axis_' + str(axis) +  '.eps'
        else:
            title = 'Logs/results/' +  TARGET_COL + "_backcast_" +str(len(backcast)) + '_step_' + str(step) + '_axis_' + str(axis)  +'.eps'
        nan_1 = np.empty(len(backcast))
        nan_1[:] = np.nan
        nan_2 = np.empty(len(prediction))
        nan_2[:] = np.nan

        observed = np.append(backcast, forecast)
        predicted = np.append(nan_1, prediction)

        data = pd.DataFrame({
            'observed': observed,
            'predicted': predicted
        })

        fig = sns.lineplot(data=data, dashes=False, palette='colorblind')
        fig.set(ylabel="Grid loss")
        fig.set(xlabel="Time")
        fig.figure.savefig(title)
        plt.close()

    if TRAINING: #takes the whole night
        for backcast_length in LOOKBACK_WINDOWS:

            model = NBeatsNet(input_dim=INPUT_DIM, exo_dim=exo_dim, backcast_length=backcast_length,
                            forecast_length=FORECAST_LENGTH,stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
                            nb_blocks_per_stack=2,thetas_dim=(4, 8), share_weights_in_stack=False,
                            hidden_layer_units=128,nb_harmonics=10)

            model.compile_model(loss='mae', learning_rate=1e-5)
            train_model(df=X_train, model=model)


    ''' Forecast '''

    print("\nForecast...")

    pred = pd.DataFrame(columns = ['144', '288', '432', '576'])

    model_144_grid =  NBeatsNet.load('nbeats_grid3_144_G.h5')
    model_288_grid =  NBeatsNet.load('nbeats_grid3_288_G.h5')
    model_432_grid =  NBeatsNet.load('nbeats_grid3_432_G.h5')
    model_576_grid =  NBeatsNet.load('nbeats_grid3_576_G.h5')

    models = [model_144_grid, model_288_grid, model_432_grid, model_576_grid]

    for i in range(len(models)):
        model = models[i]
        forecast_data = pd.concat([X_train.tail(LOOKBACK_WINDOWS[i]+FORECAST_LENGTH-1), X_test])
        x_pred, e_pred, y_pred = get_data(forecast_data, LOOKBACK_WINDOWS[i], FORECAST_LENGTH, mode="pred")
        predictions = []
        for j in tqdm(range(x_pred.shape[0])):
            x_step = np.array([x_pred[j]])
            e_step = np.array([e_pred[j]])
            y_step = np.array([y_pred[j]])
            result = models[i].predict([x_step, e_step])
            predictions.append(result[:,:,-1][:,-1]) # only append the last prediction
            model.fit([x_step, e_step], y_step, epochs=40, batch_size=1, verbose=0)
        pred[str(LOOKBACK_WINDOWS[i])] = predictions

    ''' Save predictions '''

    np.savetxt('Data/predictions/nbeats/y_pred_grid_' + str(GRID_NUMBER+1) + '_nbeats_allmodels.csv', pred, delimiter = ',')
    # the ensemble selects the mean
    mean = pred.mean(axis=1) # axis=0 to find the mean of each row
    np.savetxt('Data/predictions/nbeats/y_pred_grid_' + str(GRID_NUMBER+1) + '_nbeats.csv', mean, delimiter = ',')

if __name__ == '__main__':
    main()
