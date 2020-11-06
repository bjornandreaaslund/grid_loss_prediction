'''
'''

from pathlib import Path

from nbeats_keras.model import NBeatsNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def main():
    TIME_STEPS: int = 24
    INPUT_DIM: int = 22
    OUTPUT_DIM: int = 24
    DATASET_PATH = Path('/Data/serialized/processed_x_train_pickle')

    # Definition of the model
    model = NBeatsNet(input_dim=INPUT_DIM, backcast_length=TIME_STEPS, forecast_length=OUTPUT_DIM,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                      thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer
    model.compile_model(loss='mae', learning_rate=1e-5)

    # define the data
    data = pd.read_pickle(DATASET_PATH)

    # seperate features from target variable
    y_data = data['grid1-loss']
    x_data = data['grid1-load']

    # split data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = 0.2, shuffle=False)

    # Train the model.
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

    # Save the model for later.
    model.save('n_beats_model.h5')

    # Predict on the testing set.
    predictions = model.predict(x_test)
    print(predictions.shape)

    # Load the model.
    model2 = NBeatsNet.load('n_beats_model.h5')

    predictions2 = model2.predict(x_test)
    np.testing.assert_almost_equal(predictions, predictions2)


def demo():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

    # Definition of the model.
    model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                      thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]

    # Train the model.
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

    # Save the model for later.
    model.save('n_beats_model.h5')

    # Predict on the testing set.
    predictions = model.predict(x_test)
    print(predictions.shape)

    # Load the model.
    model2 = NBeatsNet.load('n_beats_model.h5')

    predictions2 = model2.predict(x_test)
    np.testing.assert_almost_equal(predictions, predictions2)


if __name__ == '__main__':
    main()