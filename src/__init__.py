# src/__init__.py
# This file makes src/ a Python package

from .data_loader import load_news_data, filter_by_stock
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