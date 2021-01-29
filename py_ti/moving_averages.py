import numpy as np
import pandas as pd
from check_errors import check_errors
import helper_loops


# Simple Moving Average
def sma(df, column='close', n=20, add_col=False, return_struct='numpy'):

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


# Exponential Moving Average
def ema(df, column='close', n=20, add_col=False, return_struct='numpy'):

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


# Weighted Moving Average
def wma(df, column='close', n=20, add_col=False, return_struct='numpy'):

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


# Hull Moving Average
# Use an even number with an integer square root for testing (4, 16, 36, 64)
# Possible extension: add other smoothing techniques
def hma(df, column='close', n=20, add_col=False, return_struct='numpy'):

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


# Wilders Moving Average
def wilders_ma(df, column='close', n=20, add_col=False, return_struct='numpy'):

    check_errors(df=df, column=column, n=n,
                 add_col=add_col, return_struct=return_struct)

    first_value = df[column].iloc[:n].rolling(window=n).mean().fillna(0)
    _arr = (pd.concat([first_value, df[column].iloc[n:]])).to_numpy()
    wilders = helper_loops.wilders_loop(_arr, n)

    if add_col == True:
        df[f'wilders({n})'] = wilders
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(wilders,
                            columns=[f'wilders({n})'],
                            index=df.index)
    else:
        return wilders


# Kaufman's Adaptive Moving Average
def kama(df, column='close', n_er=10, n_fast=2, n_slow=30,
         add_col=False, return_struct='numpy'):

    check_errors(df=df, column=column, n_er=n_er, n_fast=n_fast, n_slow=n_slow,
                 add_col=add_col, return_struct=return_struct)

    change = abs(df['close'] - df['close'].shift(n_er))
    vol = abs(df['close'] - df['close'].shift(1)).rolling(n_er).sum()
    er = change / vol
    fast = 2 / (n_fast + 1)
    slow = 2 / (n_slow + 1)
    sc = ((er * (fast - slow) + slow) ** 2).to_numpy()
    length = len(df)

    kama = helper_loops.kama_loop(df[column].to_numpy(), sc, n_er, length)

    if add_col == True:
        df[f'kama{n_er,n_fast,n_slow}'] = kama
        return df
    elif return_struct == 'pandas':
        return pd.DataFrame(kama,
                            columns=[f'kama({n_er},{n_fast},{n_slow})'],
                            index=df.index)
    else:
        return kama
