import numpy as np
import pandas as pd
from numba import jit
from check_errors import check_errors

# Returns
def returns(df, column='close', ret_method='simple',
            add_col=False, return_struct='numpy'):

    check_errors(df=df, column=column, ret_method=ret_method,
                 add_col=add_col, return_struct=return_struct)

    if ret_method == 'simple':
        returns = df[column].pct_change()
    elif ret_method == 'log':
        returns = np.log(df[column] / df[column].shift(1))

    if add_col == True:
        df[f'{ret_method}_ret'] = returns
        return df
    elif return_struct == 'pandas':
        return returns.to_frame(name=f'{ret_method}_ret')
    else:
        return returns.to_numpy()
