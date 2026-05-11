import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy import stats
import nltk
nltk.download('vader_lexicon', quiet=True)


def score_headlines_vader(df):
    """
    Apply VADER sentiment analysis to each headline.

    Parameters:
        df (pd.DataFrame): News DataFrame with headline column

    Returns:
        pd.DataFrame: DataFrame with compound, pos, neu, neg scores added
    """
    sia = SentimentIntensityAnalyzer()
    scores = df['headline'].apply(lambda h: sia.polarity_scores(h))

    df = df.copy()
    df['compound']      = scores.apply(lambda x: x['compound'])
    df['sentiment_pos'] = scores.apply(lambda x: x['pos'])
    df['sentiment_neu'] = scores.apply(lambda x: x['neu'])
    df['sentiment_neg'] = scores.apply(lambda x: x['neg'])
    return df


def score_headlines_textblob(df):
    """
    Apply TextBlob sentiment analysis to each headline.

    Parameters:
        df (pd.DataFrame): News DataFrame with headline column

    Returns:
        pd.DataFrame: DataFrame with polarity and subjectivity scores added
    """
    df = df.copy()
    df['tb_polarity']     = df['headline'].apply(
        lambda h: TextBlob(h).sentiment.polarity)
    df['tb_subjectivity'] = df['headline'].apply(
        lambda h: TextBlob(h).sentiment.subjectivity)
    return df


def classify_sentiment(compound_score):
    """
    Classify a compound score as Positive, Neutral or Negative.

    Parameters:
        compound_score (float): VADER compound score (-1 to +1)

    Returns:
        str: 'Positive', 'Neutral', or 'Negative'
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def align_dates(news_df, stock_df):
    """
    Align news dates to the nearest stock trading day.
    Articles published on weekends or holidays are mapped
    to the next available trading day.

    Parameters:
        news_df (pd.DataFrame): News DataFrame with date column
        stock_df (pd.DataFrame): Stock DataFrame with Date column

    Returns:
        pd.DataFrame: News DataFrame with aligned trading_date column
    """
    trading_days = pd.to_datetime(
        stock_df['Date']).dt.normalize()
    trading_days_set = sorted(trading_days.unique())

    news_df = news_df.copy()
    news_df['date_only'] = pd.to_datetime(
        news_df['date']).dt.normalize().dt.tz_localize(None)

    def get_next_trading_day(date):
        for td in trading_days_set:
            if td >= date:
                return td
        return None

    news_df['trading_date'] = news_df['date_only'].apply(
        get_next_trading_day)
    return news_df


def aggregate_daily_sentiment(news_df):
    """
    Aggregate multiple articles per day into a single
    average daily sentiment score.

    Parameters:
        news_df (pd.DataFrame): News DataFrame with trading_date
                                and compound columns

    Returns:
        pd.DataFrame: Daily sentiment DataFrame
    """
    daily = news_df.groupby('trading_date').agg(
        compound=('compound', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()

    daily['sentiment_label'] = daily['compound'].apply(
        classify_sentiment)
    return daily


def compute_daily_returns(stock_df):
    """
    Compute daily percentage change in closing prices.
    Formula: (Close_t - Close_t-1) / Close_t-1 * 100

    Parameters:
        stock_df (pd.DataFrame): Stock DataFrame with Date
                                 and Close columns

    Returns:
        pd.DataFrame: DataFrame with Date and daily_return columns
    """
    df = stock_df.copy()
    df['daily_return'] = df['Close'].pct_change() * 100
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df[['Date', 'Close', 'daily_return']].dropna()


def merge_sentiment_returns(daily_sentiment, daily_returns):
    """
    Merge daily sentiment scores with daily stock returns
    on the aligned trading date.

    Parameters:
        daily_sentiment (pd.DataFrame): Daily sentiment DataFrame
        daily_returns (pd.DataFrame): Daily returns DataFrame

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    merged = pd.merge(
        daily_sentiment,
        daily_returns,
        left_on='trading_date',
        right_on='Date',
        how='inner'
    ).dropna()
    return merged


def compute_correlation(merged_df):
    """
    Compute Pearson, Spearman and Kendall correlation coefficients
    between daily sentiment scores and daily stock returns.

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame with compound
                                  and daily_return columns

    Returns:
        dict: Dictionary with all three correlation results
    """
    pearson_r,  pearson_p  = stats.pearsonr(
        merged_df['compound'], merged_df['daily_return'])
    spearman_r, spearman_p = stats.spearmanr(
        merged_df['compound'], merged_df['daily_return'])
    kendall_r,  kendall_p  = stats.kendalltau(
        merged_df['compound'], merged_df['daily_return'])

    return {
        'pearson':  {'r': pearson_r,  'p': pearson_p},
        'spearman': {'r': spearman_r, 'p': spearman_p},
        'kendall':  {'r': kendall_r,  'p': kendall_p}
    }