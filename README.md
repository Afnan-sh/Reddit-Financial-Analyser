# Reddit vs. Market: Financial Sentiment Analysis Pipeline 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-FinBERT-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A professional-grade data analytics pipeline that processes over **100,000 Reddit posts** to quantify social sentiment and measure its correlation with 2024 stock market performance.

This project treats social media as an "Alternative Data" stream, using Natural Language Processing (FinBERT) to identify market signals, filter out noise, and detect "Alpha" users who predict market movements.

## 🚀 Key Features

* **Intelligent Data Ingestion:** Recursively parses thousands of JSON files organized by subreddit and week.
* **Noise Filtering:** Automatically removes "Advice" requests, deleted posts, and non-financial discussions using Regex and keyword heuristics.
* **Financial NLP:** Uses **FinBERT** (BERT model pre-trained on financial text) rather than generic sentiment tools for higher accuracy on market terminology.
* **Weighted Sentiment Scoring:** Implements a logarithmic weighting formula to factor in post virality (Upvotes + Comments).
* **Three-Level Analysis:**
    1.  **Macro:** Global Reddit Sentiment vs. S&P 500 (Predictive/Reactive Lag Analysis).
    2.  **Sector:** Bullish/Bearish sentiment breakdown by Subreddit.
    3.  **Micro:** Identification of "Oracle Users" (High predictive correlation with subsequent market days).

---

## 🛠️ Architecture & Methodology

The project follows a modular ETL (Extract, Transform, Load) & Analysis workflow:

### 1. The Math
We don't count every post equally. A post with 5,000 upvotes carries more weight than one with 2.
$$
\text{Weighted Score} = \text{FinBERT Score} \times \log(1 + \text{Upvotes} + \text{Comments})
$$

### 2. The Pipeline

1.  **extract (`1_scraper.py`):**
    * Uses pullpush.io.
    * Extracts Reddit posts from 8 different subreddits by week. 
2.  **Clean (`1_clean_data.py`):**
    * Iterates through `weekly_reddit_data_2024/`.
    * Extracts Tickers (`$TSLA`, `NVDA`) and Subreddit names.
    * Filters out "Help/Advice" posts to focus on market commentary.
3.  **Analyze (`3_analysis.py`):**
    * Applies FinBERT to `cleaned_reddit_data.csv`.
    * Fetches 2024 Market Data (`SPY` or `^NYA`) via `yfinance`.
    * Performs Pearson Correlation tests (Lags -1, 0, +1).
    * Generates visual plots and user leaderboards.

---

## 📂 Project Structure

```text
├── weekly_reddit_data_2024/   # Source Data (Not included in repo if large)
│   ├── dividends/
│   ├── investing/
│   └── ...
├── 1_scraper.py                # Script: Scraping ans storing in json
├── 2_clean_data.py            # Script: ETL, cleaning, and filtering
├── 3_analysis.py     # Script: AI Inference and Market Comparison
├── cleaned_reddit_data.csv    # Output: Processed dataset (Generated)
├── final_sentiment_analysis.csv # Output: Final scored dataset
├── README.md                  # Documentation
└── requirements.txt           # Python dependencies
