from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


processed_data_path = Path('Data/processed/grid_data.csv')
raw_data_path = Path('Data/raw/train.csv')


train = pd.read_csv(processed_data_path, header=0)
raw = pd.read_csv(raw_data_path)

print(raw.head())

for col in raw.columns:
    print(col)
