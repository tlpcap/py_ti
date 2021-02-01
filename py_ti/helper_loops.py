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


@jit
def psar_loop(psar, high, low, af_step, max_af):
    """
    Wilder's Parabolic Stop and Reversal Helper Loop
    Jit used to improve performance
    """

    length = len(psar)
    uptrend = True
    af = af_step
    high_point = high[0]
    low_point = low[0]
    psar_up = np.empty(length)
    psar_up.fill(np.nan)
    psar_down = np.empty(length)
    psar_down.fill(np.nan)

    for i in range(2, length):

        reversal = False

        if uptrend:
            psar[i] = psar[i-1] + af * (high_point - psar[i-1])

            if low[i] < psar[i]:
                reversal = True
                psar[i] = high_point
                low_point = low[i]
                af = af_step
            else:
                if high[i] > high_point:
                    high_point = high[i]
                    af = min(af + af_step, max_af)

                if low[i-2] < psar[i]:
                    psar[i] = low[i-2]
                elif low[i-1] < psar[i]:
                    psar[i] = low[i-1]

        else:
            psar[i] = psar[i-1] - af * (psar[i-1] - low_point)

            if high[i] > psar[i]:
                reversal = True
                psar[i] = low_point
                high_point = high[i]
                af = af_step
            else:
                if low[i] < low_point:
                    low_point = low[i]
                    af = min(af + af_step, max_af)

                if high[i-2] > psar[i]:
                    psar[i] = high[i-2]
                elif high[i-1] > psar[i]:
                    psar[i] = high[i-1]

        uptrend = uptrend ^ reversal

        if uptrend:
            psar_up[i] = psar[i]
        else:
            psar_down[i] = psar[i]

    return psar


