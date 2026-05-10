import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_headline_length(df, ticker):
    """
    Plot headline length distribution and boxplot.

    Parameters:
        df (pd.DataFrame): News DataFrame with headline_length column
        ticker (str): Stock ticker for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['headline_length'], bins=30,
                 color='steelblue', edgecolor='white')
    axes[0].set_title('Headline Length Distribution',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Character Count')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df['headline_length'].mean(), color='red',
                    linestyle='--',
                    label=f"Mean: {df['headline_length'].mean():.1f}")
    axes[0].legend()

    axes[1].boxplot(df['headline_length'], patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].set_title('Headline Length Boxplot',
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Character Count')

    plt.suptitle(f'{ticker} News Headline Length Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_publisher_activity(publisher_counts, ticker):
    """
    Plot top 10 most active publishers.

    Parameters:
        publisher_counts (pd.Series): Value counts of publishers
        ticker (str): Stock ticker for plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    publisher_counts.head(10).plot(
        kind='barh', ax=ax, color='steelblue', edgecolor='white')

    ax.set_title(f'Top 10 Most Active Publishers — {ticker}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Articles')
    ax.set_ylabel('Publisher')
    ax.invert_yaxis()

    for i, v in enumerate(publisher_counts.head(10)):
        ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_email_domains(domain_counts, ticker):
    """
    Plot email publisher domain distribution.

    Parameters:
        domain_counts (pd.DataFrame): Domain counts DataFrame
        ticker (str): Stock ticker for plot title
    """
    if domain_counts.empty:
        print(f"No email-based publishers found for {ticker}.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    domain_counts.plot(
        kind='barh', x='domain', y='count',
        ax=ax, color='steelblue', edgecolor='white', legend=False)

    ax.set_title(f'Articles by Email Publisher Domain — {ticker}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Articles')
    ax.set_ylabel('Domain')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_time_series(df, daily_counts, ticker):
    """
    Plot daily news volume and publication hour distribution.

    Parameters:
        df (pd.DataFrame): News DataFrame with hour column
        daily_counts (pd.Series): Articles per day
        ticker (str): Stock ticker for plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(daily_counts.index, daily_counts.values,
                 color='steelblue', linewidth=1.5)
    axes[0].fill_between(daily_counts.index, daily_counts.values,
                         alpha=0.3, color='steelblue')
    axes[0].set_title(f'Daily {ticker} News Volume Over Time',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Number of Articles')

    hour_counts = df['hour'].value_counts().sort_index()
    axes[1].bar(hour_counts.index, hour_counts.values,
                color='steelblue', edgecolor='white')
    axes[1].set_title('Article Publication by Hour of Day',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hour (UTC)')
    axes[1].set_ylabel('Number of Articles')
    axes[1].set_xticks(range(0, 24))

    plt.suptitle(f'{ticker} News Time Series Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_keywords(word_freq, tfidf_keywords, tfidf_scores, ticker):
    """
    Plot top keywords by frequency and TF-IDF score.

    Parameters:
        word_freq (pd.DataFrame): Word frequency DataFrame
        tfidf_keywords (array): TF-IDF feature names
        tfidf_scores (array): TF-IDF scores
        ticker (str): Stock ticker for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(word_freq['keyword'], word_freq['frequency'],
                 color='steelblue', edgecolor='white')
    axes[0].set_title('Top 20 Keywords by Frequency',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Keyword')
    axes[0].invert_yaxis()

    tfidf_df = pd.DataFrame({
        'keyword': tfidf_keywords,
        'score': tfidf_scores
    }).sort_values('score', ascending=False)

    axes[1].barh(tfidf_df['keyword'], tfidf_df['score'],
                 color='coral', edgecolor='white')
    axes[1].set_title('Top 20 Keywords by TF-IDF Score',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('TF-IDF Score')
    axes[1].set_ylabel('Keyword')
    axes[1].invert_yaxis()

    plt.suptitle(f'{ticker} News — Keyword & Topic Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_lda_topics(lda_model, feature_names, n_topics, ticker):
    """
    Plot LDA topic modeling results.

    Parameters:
        lda_model: Fitted LDA model
        feature_names (array): Vocabulary feature names
        n_topics (int): Number of topics
        ticker (str): Stock ticker for plot title
    """
    fig, axes = plt.subplots(1, n_topics, figsize=(20, 4))

    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [topic[i] for i in top_indices]

        axes[topic_idx].barh(top_words, top_scores,
                             color='steelblue', edgecolor='white')
        axes[topic_idx].set_title(f'Topic {topic_idx + 1}',
                                  fontsize=12, fontweight='bold')
        axes[topic_idx].invert_yaxis()
        axes[topic_idx].set_xlabel('Score')

    plt.suptitle(f'{ticker} — LDA Topic Modeling',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()