import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def compute_sma(series, window):
    """
    Compute Simple Moving Average using TA-Lib.

    Parameters:
        series (pd.Series): Close price series
        window (int): Number of periods

    Returns:
        np.array: SMA values
    """
    return talib.SMA(series, timeperiod=window)


def compute_ema(series, window):
    """
    Compute Exponential Moving Average using TA-Lib.

    Parameters:
        series (pd.Series): Close price series
        window (int): Number of periods

    Returns:
        np.array: EMA values
    """
    return talib.EMA(series, timeperiod=window)


def compute_rsi(series, period=14):
    """
    Compute RSI momentum indicator using TA-Lib.

    Parameters:
        series (pd.Series): Close price series
        period (int): RSI period, default 14

    Returns:
        np.array: RSI values
    """
    return talib.RSI(series, timeperiod=period)


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD, Signal Line and Histogram using TA-Lib.

    Parameters:
        series (pd.Series): Close price series
        fast (int): Fast EMA period, default 12
        slow (int): Slow EMA period, default 26
        signal (int): Signal line period, default 9

    Returns:
        tuple: (macd, signal, histogram)
    """
    return talib.MACD(
        series,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal
    )


def compute_bollinger_bands(series, period=20):
    """
    Compute Bollinger Bands using TA-Lib.

    Parameters:
        series (pd.Series): Close price series
        period (int): Period for middle band SMA, default 20

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    return talib.BBANDS(
        series,
        timeperiod=period,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )


def get_daily_return(close):
    """
    Calculate daily percentage change in closing price.

    Parameters:
        close (pd.Series): Close price series

    Returns:
        pd.Series: Daily returns
    """
    return close.pct_change()


def check_stationarity(series, title=''):
    """
    Check stationarity of a time series using:
    - Rolling mean & std visualization
    - Augmented Dickey-Fuller (ADF) test

    Parameters:
        series (pd.Series): Time series with datetime index
        title (str): Title for the plot

    Returns:
        None: Prints ADF results and shows plot
    """
    rolmean = series.rolling(window=12).mean()
    rolstd = series.rolling(window=12).std()

    plt.figure(figsize=(12, 4))
    plt.plot(series,  label='Original',     color='black',  linewidth=1.2)
    plt.plot(rolmean, label='Rolling Mean', color='red',    linewidth=1.5)
    plt.plot(rolstd,  label='Rolling Std',  color='orange', linewidth=1.5)
    plt.legend()
    plt.title(f'Rolling Mean & Std — {title}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('--- Augmented Dickey-Fuller Test ---')
    result = adfuller(series.dropna())
    print(f'ADF Statistic : {result[0]:.4f}')
    print(f'p-value       : {result[1]:.4f}')
    if result[1] <= 0.05:
        print("Conclusion: ✅ Stationary (Reject Null Hypothesis)")
    else:
        print("Conclusion: ⚠️  Non-Stationary (Fail to Reject Null Hypothesis)")