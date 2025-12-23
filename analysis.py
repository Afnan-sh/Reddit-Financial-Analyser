import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from tqdm import tqdm  # For progress bars
from scipy.stats import pearsonr

INPUT_FILE = 'cleaned_reddit_data.csv'
MARKET_TICKER = "SPY"  # S&P 500 ETF (Best proxy for general market)
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# Thresholds for User Analysis
MIN_USER_POSTS = 5        # Users must have at least 5 posts to be analyzed
MIN_USER_KARMA = 10       # Minimum total engagement (upvotes + comments)

class SentimentEngine:
    def __init__(self):
        print("Initializing FinBERT (Financial Sentiment Model)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model.to(self.device) # Move model to GPU if available
        self.model.eval() # Set to evaluation mode

    def batch_score(self, texts, batch_size=32):
        """
        Runs inference in batches to manage memory and speed.
        Returns a list of scalar scores (Positive_Prob - Negative_Prob).
        """
        scores = []
        # Loop through data in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring Sentiment"):
            batch = texts[i:i+batch_size].tolist()
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(self.device) for key, val in inputs.items()} # Move input tensors to GPU

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)

            batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy() # Move to CPU before numpy conversion
            scores.extend(batch_scores)

        return scores

# ==========================================
# 3. ANALYSIS MODULES
# ==========================================

def load_and_score_data():
    """Loads cleaned data and appends sentiment scores."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with empty text
    df = df.dropna(subset=['full_text'])

    # Initialize Engine
    engine = SentimentEngine()

    # Run Sentiment Analysis
    df['sentiment_score'] = engine.batch_score(df['full_text'])

    # Calculate Engagement Weight
    # Formula: Score * Log(1 + Upvotes + Comments)
    df['engagement'] = df['upvotes'] + df['num_comments']
    df['weighted_score'] = df['sentiment_score'] * np.log1p(df['engagement'])

    return df

def get_market_data():
    """Fetches 2024 Market Data."""
    print(f"Fetching Market Data for {MARKET_TICKER}...")
    market = yf.download(MARKET_TICKER, start=START_DATE, end=END_DATE)

    # If columns are a MultiIndex (e.g., from yfinance when auto_adjust=True or multiple tickers)
    # flatten them to use the metric name (e.g., 'Adj Close' instead of ('Adj Close', 'SPY'))
    if isinstance(market.columns, pd.MultiIndex):
        market.columns = market.columns.get_level_values(0)

    # Check if 'Adj Close' exists, otherwise use 'Close'
    close_col = 'Adj Close' if 'Adj Close' in market.columns else 'Close'

    # Calculate Daily Returns
    market['Market_Return'] = market[close_col].pct_change()

    # Normalize Market Price for plotting (Min-Max Scaling)
    market['Normalized_Price'] = (market[close_col] - market[close_col].min()) / \
                                 (market[close_col].max() - market[close_col].min())
    return market

# --- ANALYSIS 1: MARKET REFLECTION (Accumulated) ---
def analyze_accumulated_market(df, market_df):
    print("\n--- 1. Accumulated Market Analysis ---")

    # Aggregate Reddit Sentiment by Day
    daily_sentiment = df.groupby('date')['weighted_score'].mean().to_frame(name='Reddit_Sentiment')

    # Merge with Market Data
    merged = pd.merge(daily_sentiment, market_df[['Market_Return', 'Normalized_Price']],
                      left_index=True, right_index=True, how='inner')

    # Correlation Analysis
    # Lag -1: Reddit Yesterday predicts Market Today
    lag_predictive = merged['Reddit_Sentiment'].shift(1).corr(merged['Market_Return'])

    # Lag 0: Same Day correlation
    lag_coincident = merged['Reddit_Sentiment'].corr(merged['Market_Return'])

    # Lag +1: Market Yesterday explains Reddit Today (Reactive)
    lag_reactive = merged['Reddit_Sentiment'].corr(merged['Market_Return'].shift(1))

    print(f"Predictive Power (Lag -1): {lag_predictive:.4f}")
    print(f"Coincident Power (Lag 0):  {lag_coincident:.4f}")
    print(f"Reactive Power (Lag +1):   {lag_reactive:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(merged.index, merged['Reddit_Sentiment'].rolling(window=7).mean(),
             color='orange', label='Reddit Sentiment (7-Day Avg)')
    ax2.plot(merged.index, merged['Normalized_Price'],
             color='blue', linestyle='--', label=f'{MARKET_TICKER} Price')

    ax1.set_ylabel("Sentiment Score", color='orange')
    ax2.set_ylabel("Normalized Stock Price", color='blue')
    plt.title("Does Reddit Reflect the Market? (2024)")
    plt.legend(loc='upper left')
    plt.show()

    return merged

# --- ANALYSIS 2: SUBREDDIT HEALTH (Individual) ---
def analyze_subreddits(df):
    print("\n--- 2. Subreddit Sentiment Analysis ---")

    # Group by Subreddit
    sub_stats = df.groupby('subreddit').agg({
        'weighted_score': 'mean',
        'id': 'count',
        'engagement': 'mean'
    }).rename(columns={'id': 'post_count', 'engagement': 'avg_engagement'})

    # Filter for active subreddits (e.g., > 50 posts)
    sub_stats = sub_stats[sub_stats['post_count'] > 50].sort_values(by='weighted_score', ascending=False)

    print(sub_stats)

    # Visualization
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in sub_stats['weighted_score']]
    sns.barplot(x=sub_stats['weighted_score'], y=sub_stats.index, palette=colors)
    plt.title("Net Sentiment by Subreddit (Bullish vs. Bearish)")
    plt.xlabel("Average Weighted Sentiment")
    plt.show()

# --- ANALYSIS 3: ALPHA USER IDENTIFICATION ---
def analyze_users(df, market_df):
    print("\n--- 3. User Alpha Analysis ---")

    # Filter users with significant history
    user_counts = df['author'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_POSTS].index

    df_users = df[df['author'].isin(valid_users)].copy()

    results = []

    # Iterate through each top user to find correlation with market
    for user in tqdm(valid_users, desc="Analyzing Top Users"):
        user_posts = df_users[df_users['author'] == user].set_index('date')

        # We need to resample user posts to daily to match market data
        user_daily = user_posts['sentiment_score'].resample('D').mean().dropna()

        # Align with market
        user_market = pd.merge(user_daily, market_df['Market_Return'],
                               left_index=True, right_index=True, how='inner')

        if len(user_market) > 5: # Need overlap to calculate correlation
            # Calculate Predictive Correlation (User sentiment today vs Market tomorrow)
            corr, _ = pearsonr(user_market['sentiment_score'], user_market['Market_Return'].shift(-1).fillna(0))

            results.append({
                'user': user,
                'predictive_corr': corr,
                'post_count': len(user_posts),
                'avg_sentiment': user_daily.mean()
            })

    results_df = pd.DataFrame(results)

    # Top 5 "Oracle" Users (Predictive)
    top_bulls = results_df.sort_values(by='predictive_corr', ascending=False).head(5)

    # Top 5 "Contra" Users (Inverse predictors - do the opposite of what they say)
    top_bears = results_df.sort_values(by='predictive_corr', ascending=True).head(5)

    print("\nTop 'Oracle' Users (High Predictive Correlation):")
    print(top_bulls[['user', 'predictive_corr', 'post_count']])

    print("\nTop 'Inverse' Users (Consistently Wrong):")
    print(top_bears[['user', 'predictive_corr', 'post_count']])

    return results_df

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # A. Process Data
    processed_df = load_and_score_data()
    market_data = get_market_data()

    # B. Run Analyses
    analyze_accumulated_market(processed_df, market_data)
    analyze_subreddits(processed_df)
    analyze_users(processed_df, market_data)

    # C. Save Final Results
    processed_df.to_csv("final_sentiment_analysis.csv", index=False)
    print("\nAnalysis Complete. Data saved to 'final_sentiment_analysis.csv'")