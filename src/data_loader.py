import pandas as pd


def load_news_data(filepath):
    """
    Load the financial news dataset, clean and return DataFrame.

    Steps:
    - Reads the CSV file
    - Drops the unnamed index column
    - Parses the date column handling mixed timezone formats
    - Fixes inconsistent publisher names

    Parameters:
        filepath (str): Path to the raw_analyst_ratings.csv file

    Returns:
        pd.DataFrame: Cleaned news DataFrame
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Unnamed: 0'])
    df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
    df['publisher'] = df['publisher'].replace(
        'Benzinga_Newsdesk', 'Benzinga Newsdesk'
    )
    print(f"✅ Loaded {len(df)} news articles.")
    return df


def filter_by_stock(df, ticker):
    """
    Filter the news DataFrame for a specific stock ticker.

    Parameters:
        df (pd.DataFrame): Full news DataFrame
        ticker (str): Stock ticker symbol e.g. 'AAPL'

    Returns:
        pd.DataFrame: Filtered DataFrame for the given ticker
    """
    filtered = df[df['stock'] == ticker].copy().reset_index(drop=True)
    print(f"✅ Found {len(filtered)} articles for {ticker}.")
    return filtered

    def load_stock_data(filepath):
    """
    Load a stock CSV file, enforce correct data types,
    sort by date and handle missing values.

    Parameters:
        filepath (str): Path to the stock CSV file

    Returns:
        pd.DataFrame: Cleaned stock DataFrame
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Enforce correct data types (required by TA-Lib)
    df = df.astype({
        'Open'  : 'float64',
        'High'  : 'float64',
        'Low'   : 'float64',
        'Close' : 'float64',
        'Volume': 'float64'
    })

    # Handle missing values with forward-fill
    if df.isnull().values.any():
        print(f"Found {df.isnull().sum().sum()} missing values. Filling with ffill.")
        df.ffill(inplace=True)
    else:
        print(f"✅ Loaded {len(df)} rows. No missing values.")
        return df