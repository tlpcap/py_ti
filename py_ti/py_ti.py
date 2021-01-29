import numpy as np
import pandas as pd
from check_errors import check_errors
import utils

def returns(df, column='close', ret_method='simple',
            add_col=False, return_struct='numpy'):            
    """ Calculate Returns
    
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
    1. add_col=False, return_struct='numpy' returns a numpy array (default)
    2. add_col=False, return_struct='pandas' returns a new dataframe
    3. add_col=True, adds a column to the dataframe that was passed in.
    
    Note: If add_col=True the function exits and does not execute the
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

def momentum(df, column='close', n=20, add_col=False, return_struct='numpy'):
    """ Momentum
    
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
        The lookback period.
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

    mom = df[column].diff(n)

    if add_col == True:
        df[f'mom({n})'] = mom
        return df
    elif return_struct == 'pandas':
        return mom.to_frame(name=f'mom({n})')
    else:
        return mom.to_numpy()


def rate_of_change(df, column='close', n=20,
                   add_col=False, return_struct='numpy'):
    """ Rate of Change
    
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
        The lookback period.
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

    roc = df[column].diff(n) / df[column].shift(n) * 100

    if add_col == True:
        df[f'roc({n})'] = roc
        return df
    elif return_struct == 'pandas':
        return roc.to_frame(name=f'roc({n})')
    else:
        return roc.to_numpy()


def true_range(df, add_col=False, return_struct='numpy'):
    """ True Range
    
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
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

    check_errors(df=df, add_col=add_col, return_struct=return_struct)

    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift(1))
    lc = abs(df['low'] - df['close'].shift(1))
    tr = np.nanmax([hl, hc, lc], axis=0)

    if add_col == True:
        df['true_range'] = tr
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(tr, columns=['true_range'], index=df.index)
    else:
        return tr


def atr(df, n=20, ma_method='sma', add_col=False, return_struct='numpy'):
    """ Average True Range
    
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
    n : Int, optional. The default is 20.
        The lookback period.
    ma_method : String, optional.  The default is 'sma'
        The method of smoothing the True Range.  Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
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

    check_errors(df=df, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    tr = true_range(df, add_col=False, return_struct='pandas')
    tr.columns = ['close']
    
    _ma = utils.moving_average_mapper(ma_method)
    atr = _ma(tr, n=n)            

    if add_col == True:
        df[f'{ma_method}_atr({n})'] = atr
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(atr,
                            columns=[f'{ma_method}_atr({n})'],
                            index=df.index)
    else:
        return atr


def atr_percent(df, column='close', n=20, ma_method='sma',
                add_col=False, return_struct='numpy'):
    """ Average True Range Percent
    
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
    column : String, optional. The default is 'close'.
        This is the name of the column you want to use as the denominator
        of the percentage calculatioin.
    n : Int, optional. The default is 20.
        The lookback period.
    ma_method : String, optional.  The default is 'sma'
        The method of smoothing the True Range.  Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
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

    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 add_col=add_col, return_struct=return_struct)

    _atr = atr(df, n=n, ma_method=ma_method)
    atr_prcnt = (_atr / df[column]) * 100

    if add_col == True:
        df[f'atr_%({n})'] = atr_prcnt
        return df
    elif return_struct == 'pandas':
        return atr_prcnt.to_frame(name=f'atr_%({n})')
    else:
        return atr_prcnt.to_numpy()

    
def keltner_channel(df, column='close', n=20, ma_method='sma',
                    upper_factor=2.0, lower_factor=2.0,
                    add_col=False, return_struct='numpy'):
    """ Keltner Channels
    
    Parameters
    ----------
    df : Pandas DataFrame
        A Dataframe containing the columns open/high/low/close/volume
        with the index being a date.  open/high/low/close should all
        be floats.  volume should be an int.  The date index should be
        a Datetime.
    column : String, optional. The default is 'close'.
        This is the name of the column you want to use as the denominator
        of the percentage calculatioin.
    n : Int, optional. The default is 20.
        The lookback period.
    ma_method : String, optional.  The default is 'sma'
        The method of smoothing the True Range.  Available smoothing
        methods: {'sma', 'ema', 'wma', 'hma', 'wilders'}
    upper_factor : Float, optional.  The default is 2.0
        The amount by which to multiply the ATR to create the upper channel.
    lower_factor : Float, optional.  The default is 2.0
        The amount by which to multiply the ATR to create the lower channel.
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
    
    check_errors(df=df, column=column, n=n, ma_method=ma_method,
                 upper_factor=upper_factor, lower_factor=lower_factor,
                 add_col=add_col, return_struct=return_struct)

    _ma_func = utils.moving_average_mapper(ma_method)
    
    _ma = _ma_func(df, column=column, n=n)
    _atr = atr(df, n=n, ma_method=ma_method)

    keltner_upper = _ma + (_atr * upper_factor)
    keltner_lower = _ma - (_atr * lower_factor)
    keltner = np.vstack((keltner_lower, keltner_upper)).transpose()

    if add_col == True:
        df[f'kelt({n})_lower'] = keltner_lower
        df[f'kelt({n})_upper'] = keltner_upper
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(keltner,
                            columns=[f'kelt({n})_lower', f'kelt({n})_upper'],
                            index=df.index)
    else:
        return keltner
