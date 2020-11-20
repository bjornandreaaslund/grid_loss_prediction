from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():

    pickle_path = Path('Data/serialized/processed_data_pickle')
    train = pd.read_pickle(pickle_path)

    loss = train['grid1-loss']
    print(loss.describe())

    sns.lineplot(loss)


if __name__ == "__main__":
    main()
