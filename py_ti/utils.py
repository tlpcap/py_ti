import moving_averages as ma

def moving_average_mapper(moving_average):
    """
    Map input strings to functions
    """
    moving_average_funcs = {'sma': ma.sma,
                            'ema': ma.ema,
                            'wma': ma.wma,
                            'hma': ma.hma,
                            'wilders': ma.wilders_ma,
                            'kama': ma.kama}

    moving_average = moving_average_funcs[moving_average]

    return moving_average
