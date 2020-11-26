# Grid loss prediction
A power grid transports the electricity from power producers to the consumers. However, all that is produced is not delivered to the customers. Some parts of it (typically around $8\%$) are lost in either transmission or distribution. In Norway, the grid companies are responsible for reporting this \textit{grid loss} to the institutes responsible for national transmission networks. They have to nominate the expected hourly loss one day ahead so that the electricity price can be decided. If their estimates miss the target, the companies have to pay for the difference. Therefore, it is in their interest to have as accurate predictions as possible.

## Motivation
This repository is a part of a project conducted in the course TDT4173 - Machine Learning at NTNU. The task is  show that we know how to address a machine learning task and understand what it requires to train and evaluate a model. Therefore this repository includes code to prerocess raw data, train different models and different metrics to evaluate them.

## Structure

    .
    ├── Data                      # The raw data, preprocessed data, meta data and predictions
    ├── Grid_loss_prediction      # Code for preprocessing, modelling and evaluation
    ├── Log                       # Different observations
    ├── Models                    # Trained models which are used several times
    ├── Results                   # Results generated from the evaluation for all the models
    ├── Videos                    # Videos from the validation set of N-BEATS during training
    └── README.md


## How to run
1. Clone this project locally.
2. Run the preprocessing
3. Create a model
4. Evaluate the model

Below is an example on how to run the preprocessing, create a VAR model and evaluate the predictions.
```python
python Grid_loss_prediction/Data/preprocessing.py
python Grid_loss_prediction/Model/var.py
python Grid_loss_prediction/Evaluation/evaluate.py # need to set MODEL = V in the main method
```

## Dependencies

+ Python
+ Tensorflow
+ Pandas
+ sklearn
+ nbeats_keras.model.NBeatsNet
