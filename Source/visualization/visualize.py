from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


pickle_path = Path('Data/serialized/processed_data_pickle')
train = pd.read_pickle(pickle_path)

print(train.describe())
