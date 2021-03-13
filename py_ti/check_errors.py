from pandas import DataFrame

RET_METHODS = {'simple', 'log'}
MA_METHODS = {'sma', 'ema', 'wma', 'hma', 'wilders', 'kama'}
RETURN_STRUCTS = {'numpy', 'pandas'}
DDOF = {0, 1}

DF_ERR_MESSAGE = "Error: 'df' must be a Pandas DataFrame"
COLUMN_ERR_MESSAGE = "Error: 'column' must be of type str"
COLUMN_ERR_MESSAGE_2 = "Invalid Column: column name not found in dataframe"
RET_ERR_MESSAGE = f"Invalid method. Valid methods: {RET_METHODS}"
N_ERR_MESSAGE = "Error: 'n' must be of type int"
N_ER_ERR_MESSAGE = "Error: 'n_er' must be of type int"
N_FAST_ERR_MESSAGE = "Error: 'n_fast' must be of type int"
N_MED_ERR_MESSAGE = "Error: 'n_med' must be of type int"
N_SLOW_ERR_MESSAGE = "Error: 'n_slow' must be of type int"
N_K_ERR_MESSAGE = "Error: 'n_k' must be of type int"
N_D_ERR_MESSAGE = "Error: 'n_d' must be of type int"
N_MACD_ERR_MESSAGE = "Error: 'n_macd' must be of type int"
N_SUM_ERR_MESSAGE = "Error: 'n_sum' must be of type int"
FAST_ERR_MESSAGE = "Error: 'fast' must be of type int"
SLOW_ERR_MESSAGE = "Error: 'slow' must be of type int"
SIG_ERR_MESSAGE = "Error: 'sig' must be of type int"
MA_ERR_MESSAGE = f"Invalid method. Valid methods: {MA_METHODS}"
FACTOR_ERR_MESSAGE = "Error: 'factor' must be of type float"
UPPER_FACTOR_ERR_MESSAGE = "Error: 'upper_factor' must be of type float"
LOWER_FACTOR_ERR_MESSAGE = "Error: 'lower_factor' must be of type float"
DDOF_ERR_MESSAGE = "Error: 'ddof' must be 0 or 1 and of type int"
UPPER_NUM_SD_ERR_MESSAGE = "Error: 'upper_num_sd' must be of type float"
LOWER_NUM_SD_ERR_MESSAGE = "Error: 'lower_num_sd' must be of type float"
AF_STEP_ERR_MESSAGE = "Error: 'af_step' must be of type float"
MAX_AF_ERR_MESSAGE = "Error: 'max_af' must be of type float"
ADD_COL_ERR_MESSAGE = "Error: 'add_col' must be of type bool."
RETURN_STRUCTS_ERR_MESSAGE = f"Invalid return_struct. Valid return_structs: {RETURN_STRUCTS}"

def check_errors(df=None,
                 column=None,
                 ret_method=None,
                 n=None,
                 n_er=None,
                 n_fast=None,
                 n_med=None,
                 n_slow=None,
                 n_k=None,
                 n_d=None,
                 n_macd=None,
                 n_sum=None,
                 fast=None,
                 slow=None,
                 sig=None,
                 ma_method=None,
                 factor=None,
                 upper_factor=None,
                 lower_factor=None,
                 ddof=None,
                 upper_num_sd=None,
                 lower_num_sd=None,
                 af_step=None,
                 max_af=None,
                 add_col=None,
                 return_struct=None):

    if df is not None and type(df) is not DataFrame:
        raise Exception(DF_ERR_MESSAGE)

    if column is not None and type(column) is not str:
        raise Exception(COLUMN_ERR_MESSAGE)

    if column is not None and column not in df.columns:
        raise Exception(COLUMN_ERR_MESSAGE_2)

    if ret_method is not None and ret_method not in RET_METHODS:
        raise Exception(RET_ERR_MESSAGE)

    if n is not None and type(n) is not int:
        raise Exception(N_ERR_MESSAGE)

    if n_er is not None and type(n_er) is not int:
        raise Exception(N_ER_ERR_MESSAGE)

    if n_fast is not None and type(n_fast) is not int:
        raise Exception(N_FAST_ERR_MESSAGE)

    if n_med is not None and type(n_med) is not int:
        raise Exception(N_MED_ERR_MESSAGE)
        
    if n_slow is not None and type(n_slow) is not int:
        raise Exception(N_SLOW_ERR_MESSAGE)

    if n_k is not None and type(n_k) is not int:
        raise Exception(N_K_ERR_MESSAGE)

    if n_d is not None and type(n_d) is not int:
        raise Exception(N_D_ERR_MESSAGE)
    
    if n_macd is not None and type(n_macd) is not int:
        raise Exception(N_MACD_ERR_MESSAGE)

    if n_sum is not None and type(n_sum) is not int:
        raise Exception(N_SUM_ERR_MESSAGE)

    if fast is not None and type(fast) is not int:
        raise Exception(FAST_ERR_MESSAGE)

    if slow is not None and type(slow) is not int:
        raise Exception(SLOW_ERR_MESSAGE)

    if sig is not None and type(sig) is not int:
        raise Exception(SIG_ERR_MESSAGE)

    if ma_method is not None and ma_method not in MA_METHODS:
        raise Exception(MA_ERR_MESSAGE)

    if factor is not None and type(factor) is not float:
        raise Exception(FACTOR_ERR_MESSAGE)

    if upper_factor is not None and type(upper_factor) is not float:
        raise Exception(UPPER_FACTOR_ERR_MESSAGE)

    if lower_factor is not None and type(lower_factor) is not float:
        raise Exception(LOWER_FACTOR_ERR_MESSAGE)

    if ddof is not None and ddof not in DDOF:
        raise Exception(DDOF_ERR_MESSAGE)

    if upper_num_sd is not None and type(upper_num_sd) is not float:
        raise Exception(UPPER_NUM_SD_ERR_MESSAGE)

    if lower_num_sd is not None and type(lower_num_sd) is not float:
        raise Exception(LOWER_NUM_SD_ERR_MESSAGE)

    if af_step is not None and type(af_step) is not float:
        raise Exception(AF_STEP_ERR_MESSAGE)

    if max_af is not None and type(max_af) is not float:
        raise Exception(MAX_AF_ERR_MESSAGE)
        
    if add_col is not None and type(add_col) is not bool:
        raise Exception(ADD_COL_ERR_MESSAGE)

    if return_struct is not None and return_struct not in RETURN_STRUCTS:
        raise Exception(RETURN_STRUCTS_ERR_MESSAGE)
