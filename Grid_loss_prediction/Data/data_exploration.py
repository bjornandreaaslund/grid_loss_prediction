'''
This module creates different datafram with different groups of data,
and contains code for descriptive statistics and functions for plotting.

Contents:

- General settings
- Grouping data and saving the in Data/processed
- Descriptives before cleaning
- Visualize data

'''

import os
import gc
import datetime as datetime
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import groupby, count


def main():


    ''' General Settings '''

    LOADDIR = Path('Data/').resolve()
    SAVEDIR_LOG = Path('Log/').resolve()
    SAVEDIR_META = Path('Data/meta/').resolve()
    if not os.path.exists(SAVEDIR_LOG):
        os.mkdir(os.getcwd() + SAVEDIR_LOG)
    if not os.path.exists(SAVEDIR_META):
        os.mkdir(os.path.join(os.getcwd(), SAVEDIR_META))

    DF_COLUMNS = {  'Grid_data' : ['Unnamed: 0', 'demand', 'grid1-load', 'grid1-loss','grid1-temp', 'grid2-load', 'grid2-loss','grid2_1-temp',
                                    'grid2_2-temp', 'grid3-load', 'grid3-loss', 'grid3-temp', 'has incorrect data'],
                    'Prophet' : ['grid1-loss-prophet-daily', 'grid1-loss-prophet-pred', 'grid1-loss-prophet-trend', 'grid1-loss-prophet-weekly',
                                'grid1-loss-prophet-yearly', 'grid2-loss-prophet-daily', 'grid2-loss-prophet-pred','grid2-loss-prophet-trend',
                                'grid2-loss-prophet-weekly','grid2-loss-prophet-yearly', 'grid3-loss-prophet-pred', 'grid3-loss-prophet-trend',
                                'grid3-loss-prophet-weekly', 'grid3-loss-prophet-yearly'],
                    'Seasonal' : ['season_x', 'season_y', 'month_x', 'month_y', 'week_x', 'week_y',
                                'weekday_x', 'weekday_y', 'holiday', 'hour_x', 'hour_y'],
                    'Temperature' : ['grid1-temp', 'grid2_1-temp', 'grid2_2-temp', 'grid3-temp'],
                    'Timestamp' : ['Unnamed: 0']
                }

    train = pd.read_csv(LOADDIR.joinpath('raw/train.csv'), header=0)


    ''' Grouping data and saving the in Data/processed '''

    grid_data = train[DF_COLUMNS['Grid_data']]
    grid_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    grid_data.to_csv(LOADDIR.joinpath('processed/grid_data.csv'))

    prophet = train[DF_COLUMNS['Prophet']]
    prophet.to_csv(LOADDIR.joinpath('processed/prophet_data.csv'))

    seasonal = train[DF_COLUMNS['Seasonal']]
    seasonal.to_csv(LOADDIR.joinpath('processed/seasonal_data.csv'))

    del prophet, seasonal


    ''' Descriptives before cleaning '''

    # Number of rows and columns
    print("Rows:              ", train.shape[0])
    print("Columns:           ", train.shape[1])

    # Overview of numerical columns
    # TODO: describe appropirate groups of columns
    grid_data.describe()
    grid_data.describe().T.to_latex(SAVEDIR_META.joinpath('descriptives_raw.tex'), float_format='%.2f')
    pd.DataFrame(grid_data.shape[1]*[''], index=grid_data.columns, columns=['Description']).to_latex(savedir_meta.joinpath('features_raw.tex'))

    print("Rows with NaN demand:")
    print(grid_data[grid_data['demand'].isna()], '\n')

    print("Number of nan values:")
    print(grid_data.isnull().sum(), '\n')

    # see the ranges of incorrect data
    print("There are ", train['has incorrect data'].value_counts().iloc[1], " rows with incorrect data\n")
    index_list = np.array(train.index[train['has incorrect data'] == True].tolist())

    def as_range(iterable): # not sure how to do this part elegantly
        l = list(iterable)
        if len(l) > 1:
            return '{0}-{1}'.format(l[0], l[-1])
        else:
            return '{0}'.format(l[0])

    # print out the ranges of incorrect data
    index_ranges = ','.join(as_range(g) for _, g in groupby(index_list, key=lambda n, c=count(): n-next(c)))
    print("The incorrect data is located in these ranges:")
    for index_range in index_ranges.split(','):
        print(index_range)
    print('\n')

    ''' Visualize data '''

    train = train.set_index('Unnamed: 0')

    def evaluate_seasonaldata():
        for col in DF_COLUMNS['Seasonal']:
            print(train[col].unique())
            print(col, ' has ', len(train[col].unique()), 'unique values.')
            plt.plot(train[col])
            plt.title(col)
            plt.savefig(SAVEDIR_LOG+col)
            plt.show()

    def print_corr_matrix():
        sns.clustermap(train[DF_COLUMNS['Grid_data']].corr())
        plt.title("Correlation matrix")
        plt.savefig(SAVEDIR_LOG+"corr_matrix")
        plt.show()

    def plot_wrong_sensordata():
        plt.plot(grid_data['grid2-loss'])
        plt.title('Wrong measurement for grid2 loss')
        plt.xlim(sensor_error_start-100, sensor_error_end+100)
        plt.savefig(SAVEDIR_LOG+'wrong_measurement_grid2_loss')
        plt.show()

    def plot_temperature():
        plt.plot(train[DF_COLUMNS['Temperature']])
        plt.title('The temperature for each of the grids.')
        plt.xlabel('Time')
        plt.ylabel('Temperature [K]')
        plt.savefig(SAVEDIR_LOG+'temperature')
        plt.show()

    def plot_loss():
        plt.plot(train[['grid1-loss', 'grid2-loss', 'grid2-loss']])
        plt.title('The loss from each grid.')
        plt.xlabel('Time')
        plt.ylabel('Energy [MWh]')
        plt.savefig(SAVEDIR_LOG+'loss')
        plt.show()

    def plot_load():
        plt.plot(train[['grid1-load', 'grid2-load', 'grid2-load']])
        plt.title('The load from each grid.')
        plt.xlabel('Time')
        plt.ylabel('Energy [MWh]')
        plt.savefig(SAVEDIR_LOG+'load')
        plt.show()


if __name__ == "__main__":
    main()
