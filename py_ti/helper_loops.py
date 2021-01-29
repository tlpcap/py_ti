import numpy as np
from numba import jit


@jit
def wilders_loop(data, n):
    """
    Wilder's Moving Average Helper Loop
    Jit used to improve performance
    """

    for i in range(n, len(data)):
        data[i] = (data[i-1] * (n-1) + data[i]) / n
    return data


@jit
def kama_loop(data, sc, n_er, length):
    """
    Kaufman's Adaptive Moving Average Helper Loop
    Jit used to improve performance
    """

    kama = np.full(length, np.nan)
    kama[n_er-1] = data[n_er-1]

    for i in range(n_er, length):
        kama[i] = kama[i-1] + sc[i] * (data[i] - kama[i-1])
    return kama
