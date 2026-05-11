# src/__init__.py
# This file makes src/ a Python package
# allowing modular imports across all notebooks

from .data_loader import (
    load_news_data,
    filter_by_stock,
    load_stock_data
)

from .text_analyzer import (
    clean_headline,
    get_tfidf_keywords,
    get_word_frequency,
    get_lda_topics,
    extract_email_domains
)

from .visualizer import (
    plot_headline_length,
    plot_publisher_activity,
    plot_email_domains,
    plot_time_series,
    plot_keywords,
    plot_lda_topics
)

from .analysis import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    get_daily_return,
    check_stationarity
)

from .sentiment_analyzer import (
    score_headlines_vader,
    score_headlines_textblob,
    classify_sentiment,
    align_dates,
    aggregate_daily_sentiment,
    compute_daily_returns,
    merge_sentiment_returns,
    compute_correlation
)