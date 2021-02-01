# py_ti
A collection of 21 technical indicators. Suggestions are welcome.

# Current List:<br />
Simple Returns<br />
Log Returns<br />
Historical Volatility<br />
Simple Moving Average<br />
Exponential Moving Average<br />
Weighted Moving Average<br />
Hull Moving Average<br />
Wilder's Moving Average<br />
Kaufman's Adaptive Moving Average<br />
Momentum<br />
Rate of Change<br />
True Range<br />
Average True Range<br />
Average True Range Percent<br />
Keltner Channels<br />
Bollinger Bands<br />
Relative Strength Index<br />
True Strength Index<br />
Average Directional Index<br />
Parabolic Stop-and-Reverse<br />
Supertrend<br />

# Data
Data should be in open/high/low/close/volume format in a Pandas DataFrame with the date as the index.<br />
ohlc = float<br />
volume = int<br />
date = Datetime<br />

Data Example:  
![data_example](https://user-images.githubusercontent.com/29778401/105869496-4b36a300-5fc5-11eb-8324-aaa0fc98f37d.png)

# Versions used:
python 3.8.5<br />
numpy 1.19.2<br />
pandas 1.2.1<br />
numba 0.51.2<br />
