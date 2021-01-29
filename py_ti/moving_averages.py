import numpy as np
import pandas as pd
from check_errors import check_errors
from helper_loops import wilders_loop, kama_loop


def sma(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Simple Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
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


def ema(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Exponential Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """
    
    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    first_value = df[column].iloc[:n].rolling(window=n).mean()
    _ema = pd.concat([first_value, df[column][n:]])
    ema = _ema.ewm(span=n, adjust=False).mean()

    if add_col == True:
        df[f'ema({n})'] = ema
        return df
    elif return_struct == 'pandas':
        return ema.to_frame(name=f'ema({n})')
    else:
        return ema.to_numpy()


def wma(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Weighted Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """
    
    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    weights = np.arange(1, n + 1, 1)
    wma = df[column].rolling(n).apply(lambda x: np.dot(x, weights) /
                                      weights.sum(), raw=True)

    if add_col == True:
        df[f'wma({n})'] = wma
        return df
    elif return_struct == 'pandas':
        return wma.to_frame(name=f'wma({n})')
    else:
        return wma.to_numpy()


def hma(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Hull Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """
    
    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    wma_1 = wma(df, column=column, n=n//2)
    wma_2 = wma(df, column=column, n=n)
    _df = pd.DataFrame(2 * wma_1 - wma_2, columns=[column])
    hma = wma(_df, column=column, n=int(n ** 0.5))

    if add_col == True:
        df[f'hma({n})'] = hma
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(hma, columns=[f'hma({n})'], index=df.index)
    else:
        return hma


def wilders_ma(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Wilder's Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """
    
    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    first_value = df[column].iloc[:n].rolling(window=n).mean().fillna(0)
    _arr = (pd.concat([first_value, df[column].iloc[n:]])).to_numpy()
    wilders = wilders_loop(_arr, n)

    if add_col == True:
        df[f'wilders({n})'] = wilders
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(wilders,
                            columns=[f'wilders({n})'],
                            index=df.index)
    else:
        return wilders


def kama(df, column='close', n_er=10, n_fast=2, n_slow=30,
         add_col=False, return_struct='numpy'):
    """ Kaufman's Moving Average
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
    return_struct parameter.
    """
    
    check_errors(df=df, column=column, n_er=n_er, n_fast=n_fast, n_slow=n_slow,
                 add_col=add_col, return_struct=return_struct)

    change = abs(df['close'] - df['close'].shift(n_er))
    vol = abs(df['close'] - df['close'].shift(1)).rolling(n_er).sum()
    er = change / vol
    fast = 2 / (n_fast + 1)
    slow = 2 / (n_slow + 1)
    sc = ((er * (fast - slow) + slow) ** 2).to_numpy()
    length = len(df)

    kama = kama_loop(df[column].to_numpy(), sc, n_er, length)

    if add_col == True:
        df[f'kama{n_er,n_fast,n_slow}'] = kama
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(kama,
                            columns=[f'kama({n_er},{n_fast},{n_slow})'],
                            index=df.index)
    else:
        return kama