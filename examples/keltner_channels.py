import pandas as pd
import numpy as np
import py_ti as ti

# My data is already formatted as described in the README
data = pd.read_csv('spy_data.csv')

# Return a Numpy array with 2 columns.
# Column 0 is the lower Keltner Channel
# Column 1 is the upper Keltner Channel
keltner_numpy_array = ti.keltner_channel(data) # same as ti.keltner_channel(data, n=20, add_col=False, return_struct='numpy')

# Return a new Pandas Dataframe with 2 columns.
# 1 column for each channel
keltner_dataframe = ti.keltner_channel(data, return_struct='pandas')  # same as ti.keltner_channel(data, n=20, add_col=False, return_struct='pandas')

# add 20 period Keltner Channels with SMA smoothing (default) to the Dataframe that was passed in
ti.keltner_channel(data, add_col=True) # same as ti.keltner_channel(data, n=20, add_col=True)

# add 14 period Keltner Channels with Wilders MA smoothing to the Dataframe that was passed in
ti.keltner_channels(data, n=14, ma_method='wilders', add_col=True)
