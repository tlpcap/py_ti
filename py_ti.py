import numpy as np
import pandas as pd
from numba import jit
from check_errors import check_errors

# Returns
def returns(df, column='close', ret_method='simple',
            add_col=False, return_struct='numpy'):
    """
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
    column : String, optional. The default is 'close'.
        This is the name of the column you want to operate on.
    ret_method : String, optional. The default is 'simple'.
        The kind of returns you want returned: 'simple' or 'log'
    add_col : Boolean, optional. The default is False.
        By default the function will return a numpy array. If set to
        True, the function will add a column to the dataframe that was
        passed in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'.
        Only two values accepted: 'numpy' and 'pandas'.  If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note that if add_col=True the function exits and does not execute the
    return_struct parameter.
    """

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


# Simple Moving Average
def sma(df, column='close', n=20, add_col=False, return_struct='numpy'):
"""
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
    column : String, optional. The default is 'close'.
        This is the name of the column you want to operate on.
    n : Int, optional. The default is 20.
        The lookback period for the moving average.
    add_col : Boolean, optional. The default is False.
        By default the function will return a numpy array. If set to True,
        the function will add a column to the dataframe that was passed
        in to it instead or returning a numpy array.
    return_struct : String, optional. The default is 'numpy'.
        Only two values accepted: 'numpy' and 'pandas'.  If set to
        'pandas', a new dataframe will be returned.

    Returns
    -------
    There are 3 ways to return values from this function:
    1. add_col=False, return_struct='numpy' returns numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note that if add_col=True the function exits and does not execute the
    return_struct parameter.
    """

    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    sma = df[column].rolling(window=n).mean()

    if add_col == True:
        df[f'sma({n})'] = sma
        return df
    elif return_struct == 'pandas':
        return sma.to_frame(name=f'sma({n})')
    else:
        return sma.to_numpy()
