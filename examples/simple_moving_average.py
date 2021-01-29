import pandas as pd
import numpy as np
import py_ti as ti

# My data is already formatted as described in the README
data = pd.read_csv('spy_data.csv')

# Return a Numpy array with 20 period SMA(default)
sma_numpy_array = ti.sma(data)  # same as ti.sma(data, n=20, add_col=False, return_struct='numpy')

# Return a new Pandas Dataframe with 20 period SMA
sma_dataframe = ti.sma(data, return_struct='pandas')  # same as ti.sma(data, n=20, add_col=False, return_struct='pandas')

# add a 20 period SMA column to the Dataframe that was passed in
ti.sma(data, add_col=True) # same as ti.sma(data, n=20, add_col=True)

# add a 50 period SMA column to the Dataframe that was passed in
ti.sma(data, n=50, add_col=True)
