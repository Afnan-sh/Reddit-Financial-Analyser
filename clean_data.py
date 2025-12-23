import json
import pandas as pd
import re
import glob
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DATA_DIR = 'weekly_reddit_data_2024'
OUTPUT_FILE = 'cleaned_reddit_data.csv'

BLACKLIST = set([
    'A', 'I', 'THE', 'AND', 'FOR', 'OF', 'TO', 'IN', 'IS', 'ON', 'IT', 'MY',
    'AT', 'WE', 'DO', 'BE', 'BY', 'OR', 'AS', 'IF', 'SO', 'UP', 'NO', 'GO',
    'ME', 'DD', 'CEO', 'CFO', 'CTO', 'ATH', 'ATL', 'GDP', 'IRS', 'SEC', 'ETF',
    'IRA', '401K', 'YOLO', 'FOMO', 'ROTH', 'USA', 'USD', 'EDIT', 'TLDR', 'RH'
])

STOP_WORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off','again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'most', 'other', 'some', 'such', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'I', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

KEYWORDS = [
    'market', 'stock', 'dividend', 'invest', 'yield', 'portfolio', 'crash',
    'rally', 'bull', 'bear', 'recession', 'inflation', 'fed', 'interest',
    'spy', 'voo', 'qqq', 'dow', 'nasdaq', 'nyse'
]

# ==========================================
# 2. FUNCTIONS
# ==========================================
def clean_text(text):
    """
    Cleans the text by removing punctuation, stop words, and consecutive repeated words.
    """
    if not isinstance(text, str) or text.strip() == '':
        return ''
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    # Remove stop words
    words = [w for w in words if w not in STOP_WORDS]
    # Remove consecutive duplicates
    cleaned_words = []
    prev = None
    for w in words:
        if w != prev:
            cleaned_words.append(w)
        prev = w
    # Join back
    return ' '.join(cleaned_words)

def extract_tickers(text):
    text = str(text)
    explicit = re.findall(r'\$([A-Z]{1,5})\b', text)
    potential = re.findall(r'\b([A-Z]{2,5})\b', text)
    candidates = set(explicit + potential)
    return [t for t in candidates if t not in BLACKLIST]

def is_valid_post(row):
    """
    Returns False if the post should be filtered out.
    Criteria:
    1. Body is [deleted]/[removed] AND title is too short.
    2. Title or Body contains the word 'advice'.
    """
    title = str(row['title'])
    body = str(row['body'])
    combined_text_lower = (title + " " + body).lower()

    # Filter 1: Remove "Advice" requests (e.g., "Need advice on my portfolio")
    if 'advice' in combined_text_lower:
        return False

    # Filter 2: Remove deleted/removed posts with no useful title
    if body in ['[deleted]', '[removed]'] and len(title.split()) < 5:
        return False
        
    return True

def process_file(file_path):
    try:
        # Extract subreddit from filename (e.g., dividends_W1_2024... -> dividends)
        filename = os.path.basename(file_path)
        subreddit_name = filename.split('_')[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # Add metadata
        df['subreddit'] = subreddit_name
        df['date'] = pd.to_datetime(df['timestamp_utc'], unit='s').dt.date
        
        # --- NEW FILTERING LOGIC ---
        df['is_valid'] = df.apply(is_valid_post, axis=1)
        df = df[df['is_valid']].copy()

        # Combine Text
        df['full_text'] = df['title'] + " " + df['body'].replace({'\[deleted\]': '', '\[removed\]': ''}, regex=True)

        # Clean the text: remove repeated words and common words
        df['full_text'] = df['full_text'].apply(clean_text)

        return df
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    files = glob.glob(os.path.join(RAW_DATA_DIR, "**/*.json"), recursive=True)
    print(f"Found {len(files)} JSON files.")

    all_dfs = [process_file(f) for f in files]
    
    valid_dfs = [df for df in all_dfs if not df.empty]
    
    if not valid_dfs:
        print("No valid data loaded.")
    else:
        full_df = pd.concat(valid_dfs, ignore_index=True)
        print(f"Raw Posts: {len(full_df)}")

        full_df['tickers'] = full_df['full_text'].apply(extract_tickers)
        
        mask_keywords = full_df['full_text'].str.contains('|'.join(KEYWORDS), case=False, na=False)
        mask_tickers = full_df['tickers'].apply(lambda x: len(x) > 0)
        
        final_df = full_df[mask_keywords | mask_tickers].copy()
        print(f"Relevant Posts (Advice Filtered): {len(final_df)}")

        # Save to CSV
        final_df['tickers'] = final_df['tickers'].apply(lambda x: ','.join(x))
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved processed data to {OUTPUT_FILE}")