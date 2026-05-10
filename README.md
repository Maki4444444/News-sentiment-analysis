# News Sentiment Analysis — Predicting Price Moves with Financial Headlines

## Overview
This repository contains a rigorous analytical pipeline built for **Nova Financial Solutions** that connects financial news sentiment to stock price movements. The project quantifies sentiment in financial news headlines using NLP techniques and measures the statistical relationship between sentiment scores and daily stock returns across five major technology stocks: **AAPL, AMZN, GOOG, FB (Meta), and NVDA**.

## Business Objective
Nova Financial Solutions aims to enhance its predictive analytics capabilities by building a data-driven pipeline that:
1. **Sentiment Analysis** — Applies NLP techniques to quantify the tone expressed in financial news headlines and associates sentiment scores with stock symbols
2. **Correlation Analysis** — Measures the statistical relationship between news sentiment and daily stock price movements to inform actionable investment strategies

## Repository Structure
News-sentiment-analysis/
├── .github/
│   └── workflows/
│       └── unittests.yml        ← CI/CD pipeline
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/                     ← Stock CSVs and news dataset (gitignored)
├── notebooks/
│   ├── AAPL_eda.ipynb           ← Task 1: EDA for Apple
│   ├── AMZN_eda.ipynb           ← Task 1: EDA for Amazon
│   ├── GOOG_eda.ipynb           ← Task 1: EDA for Google
│   ├── FB_eda.ipynb             ← Task 1: EDA for Meta (Facebook)
│   ├── NVDA_eda.ipynb           ← Task 1: EDA for NVIDIA
│   ├── AAPL_technical_analysis.ipynb  ← Task 2: Technical Analysis for Apple
│   ├── AMZN_technical_analysis.ipynb  ← Task 2: Technical Analysis for Amazon
│   ├── GOOG_technical_analysis.ipynb  ← Task 2: Technical Analysis for Google
│   ├── FB_technical_analysis.ipynb    ← Task 2: Technical Analysis for Meta
│   └── NVDA_technical_analysis.ipynb  ← Task 2: Technical Analysis for NVIDIA
├── src/
│   ├── init.py
│   ├── data_loader.py           ← Load and clean news & stock data
│   ├── text_analyzer.py         ← NLP: TF-IDF, CountVectorizer, LDA
│   ├── visualizer.py            ← All EDA plotting functions
│   └── analysis.py              ← Technical indicators & stationarity
├── tests/
│   └── test_placeholder.py      ← CI/CD unit tests
└── scripts/
└── README.md
## Datasets
- **Financial News Dataset** — `raw_analyst_ratings.csv`: 1.4M+ financial news headlines with publisher, date, and stock symbol (FNSPID dataset)
- **Stock Price Data** — Individual CSV files for AAPL, AMZN, GOOG, FB, NVDA containing OHLCV data from 2009–2023

> Note: Data files are excluded from version control due to size. Download from the course Google Drive and place in `data/raw/`.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Maki4444444/News-sentiment-analysis.git
cd News-sentiment-analysis
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place data files
Download `newsData.zip` and `yfinance_data.zip` from the course Google Drive, extract and place all CSV files in `data/raw/`

### 5. Launch notebooks
```bash
jupyter notebook
```
Navigate to `notebooks/` and open any EDA or technical analysis notebook.

## Tasks

### Task 1 — Exploratory Data Analysis (EDA)
Each company has a dedicated EDA notebook covering:
- **Descriptive Statistics** — headline character count distributions, sample analysis
- **Publisher Analysis** — most active publishers, email domain extraction
- **Time Series Analysis** — daily news volume trends, publication hour patterns
- **Text Analysis** — TF-IDF keywords, CountVectorizer frequency, LDA topic modeling

### Task 2 — Technical Analysis
Each company has a dedicated technical analysis notebook covering:
- **Data Preparation** — type enforcement, missing value handling with ffill
- **Moving Averages** — SMA and EMA (20-day, 50-day) using TA-Lib
- **Bollinger Bands** — price channel analysis
- **RSI** — overbought/oversold momentum indicator
- **MACD** — momentum shift and trend reversal detection
- **PyNance Indicators** — SMA cross-validation and additional metrics
- **QuantStats Metrics** — Sharpe Ratio, Max Drawdown, CAGR, Volatility
- **Time Series Decomposition** — Trend, Seasonal, Residual components
- **Stationarity Analysis** — ADF test and first-order differencing

### Task 3 — Sentiment Correlation (Coming Soon)
- Date alignment between news and stock datasets
- VADER sentiment scoring of headlines
- Daily return calculation
- Pearson correlation analysis between sentiment and returns

## CI/CD
GitHub Actions runs unit tests automatically on every push to `main`, `task-1`, `task-2`, and `task-3` branches. Tests are located in `tests/`.

## Branch Strategy
| Branch | Purpose |
|--------|---------|
| `main` | Stable, merged code |
| `task-1` | EDA notebooks and src modules |
| `task-2` | Technical analysis notebooks |
| `task-3` | Sentiment correlation (upcoming) |

## Dependencies
Key packages used:
- `pandas`, `numpy` — data manipulation
- `TA-Lib` — technical indicators
- `pynance` — additional financial metrics
- `quantstats` — performance metrics
- `statsmodels` — time series decomposition and stationarity
- `nltk`, `scikit-learn` — NLP and text analysis
- `matplotlib`, `seaborn` — visualization
- `vaderSentiment`, `textblob` — sentiment analysis

See `requirements.txt` for full list with pinned versions.

## Author
Developed as part of the 10 Academy KAIM Week 1 Challenge.