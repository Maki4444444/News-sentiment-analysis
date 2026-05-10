import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def clean_headline(text):
    """
    Clean a headline by lowercasing, removing punctuation,
    and stripping stopwords.

    Parameters:
        text (str): Raw headline text

    Returns:
        str: Cleaned headline
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def get_tfidf_keywords(corpus, max_features=20, ngram_range=(1, 2)):
    """
    Compute TF-IDF keywords from a list of cleaned headlines.

    Parameters:
        corpus (pd.Series): Series of cleaned headlines
        max_features (int): Number of top keywords
        ngram_range (tuple): Range of n-grams

    Returns:
        tuple: (tfidf_matrix, feature_names, tfidf_scores)
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().mean(axis=0)
    return tfidf_matrix, feature_names, tfidf_scores


def get_word_frequency(corpus, max_features=20, ngram_range=(1, 2)):
    """
    Compute word frequency from a list of cleaned headlines.

    Parameters:
        corpus (pd.Series): Series of cleaned headlines
        max_features (int): Number of top keywords
        ngram_range (tuple): Range of n-grams

    Returns:
        pd.DataFrame: DataFrame with keyword and frequency columns
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    word_matrix = vectorizer.fit_transform(corpus)
    word_freq = pd.DataFrame({
        'keyword': vectorizer.get_feature_names_out(),
        'frequency': word_matrix.toarray().sum(axis=0)
    }).sort_values('frequency', ascending=False)
    return word_freq


def get_lda_topics(corpus, n_topics=5, max_features=1000):
    """
    Perform LDA topic modeling on a corpus of cleaned headlines.

    Parameters:
        corpus (pd.Series): Series of cleaned headlines
        n_topics (int): Number of topics to extract
        max_features (int): Vocabulary size

    Returns:
        tuple: (lda_model, feature_names, topics_list)
    """
    lda_vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 1))
    lda_matrix = lda_vectorizer.fit_transform(corpus)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10
    )
    lda_model.fit(lda_matrix)

    feature_names = lda_vectorizer.get_feature_names_out()
    topics = []
    for topic in lda_model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(top_words)

    return lda_model, feature_names, topics


def extract_email_domains(df):
    """
    Extract domains from email-based publisher names.

    Parameters:
        df (pd.DataFrame): News DataFrame with publisher column

    Returns:
        pd.DataFrame: DataFrame with domain counts
    """
    email_pubs = df[df['publisher'].str.contains('@', na=False)].copy()
    if len(email_pubs) == 0:
        return pd.DataFrame()
    email_pubs['domain'] = email_pubs['publisher'].str.extract(r'@([\w.]+)')
    return email_pubs['domain'].value_counts().reset_index()