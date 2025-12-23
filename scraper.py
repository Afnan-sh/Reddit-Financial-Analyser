import argparse
import datetime
import json
import os
import time
import glob

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Defaults (override via CLI args) ---
DEFAULT_SUBREDDIT = "wallstreetbets"
DEFAULT_YEAR = 2024

PULLPUSH_BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

# PullPush max is 100
MAX_SIZE = 100

# Safety limits (override via CLI args)
DEFAULT_MAX_POSTS_PER_WEEK = 1000
DEFAULT_MAX_REQUESTS_PER_WEEK = 250
DEFAULT_SLEEP_SECONDS = 1.2
DEFAULT_TIMEOUT_SECONDS = 30

# Retry policy for transient network/DNS/HTTP issues
DEFAULT_MAX_RETRIES = 6

# --- Utility Functions ---

def get_week_timestamps(year, iso_week_num):
    """Calculates UNIX timestamps for the start/end of an ISO week (Mon..Mon) in UTC."""
    start_date = datetime.date.fromisocalendar(year, iso_week_num, 1)  # Monday
    start_dt = datetime.datetime.combine(start_date, datetime.time(0, 0, 0), tzinfo=datetime.timezone.utc)
    end_dt = start_dt + datetime.timedelta(weeks=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def create_session(max_retries: int) -> requests.Session:
    """Create a requests session with retry/backoff for transient failures."""
    session = requests.Session()

    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def is_text_only_post(post):
    """
    Checks if a post is a text-only post, excluding most media/links.
    """
    # Must have a non-empty body of text (selftext)
    has_text = bool(post.get('selftext')) and post.get('selftext').lower() not in ['[removed]', '[deleted]', '']
    
    # Must not be an API-flagged video
    is_not_video = not post.get('is_video', False) 
    
    # Exclude common image/video domains
    domain = post.get('domain', '').lower()
    is_not_image_domain = not any(d in domain for d in ['i.redd.it', 'imgur', 'gfycat', 'v.redd.it', 'youtube', 'youtu.be'])
    
    return has_text and is_not_video and is_not_image_domain

def fetch_weekly_data(
    session: requests.Session,
    subreddit: str,
    week_num: int,
    week_start_ts: int,
    week_end_ts: int,
    max_posts_per_week: int,
    max_requests_per_week: int,
    sleep_seconds: float,
    timeout_seconds: int,
):
    """
    Fetches the largest possible pool of high-score posts for a given week.
    """
    current_before_ts = week_end_ts
    all_collected_posts = []
    seen_ids: set[str] = set()

    # include `id` so we can de-duplicate across pages
    FILTER_FIELDS = "id,title,selftext,score,num_comments,link_flair_text,created_utc,domain,is_video,author"

    print(f"  -> Fetching data for Week {week_num} ({datetime.datetime.utcfromtimestamp(week_start_ts).strftime('%Y-%m-%d')} to {datetime.datetime.utcfromtimestamp(week_end_ts).strftime('%Y-%m-%d')} UTC)")

    consecutive_failures = 0

    for i in range(max_requests_per_week):
        # Stop fetching once we have a large buffer to filter/sort from
        if len(all_collected_posts) >= max_posts_per_week * 3:
            break
            
        params = {
            "subreddit": subreddit,
            "after": week_start_ts,
            "before": current_before_ts,
            "size": MAX_SIZE,
            # IMPORTANT: paginate by time, not by score
            "sort": "desc",
            "sort_type": "created_utc",
            "filter": FILTER_FIELDS,
        }

        try:
            response = session.get(PULLPUSH_BASE_URL, params=params, timeout=timeout_seconds)
            # If PullPush returns a non-200, keep going; retries may already have been attempted.
            if response.status_code != 200:
                consecutive_failures += 1
                print(
                    f"  -> HTTP {response.status_code} for week {week_num} (attempt {i + 1}/{max_requests_per_week})."
                )
                if consecutive_failures >= 5:
                    print(f"  -> Too many consecutive failures for week {week_num}; skipping week.")
                    break
                time.sleep(min(30, sleep_seconds * (2 ** min(consecutive_failures, 4))))
                continue

            data = response.json()
            batch_posts = data.get("data", [])

            if not batch_posts:
                break

            consecutive_failures = 0

            # De-duplicate across pages
            new_posts = 0
            for p in batch_posts:
                pid = p.get("id")
                if not pid:
                    # fall back to a composite key if id missing
                    pid = f"{p.get('created_utc')}|{p.get('author')}|{p.get('title')}"
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                all_collected_posts.append(p)
                new_posts += 1

            # Update cursor for next request (Pagination by oldest created_utc in this batch)
            oldest_ts = min(p.get("created_utc", current_before_ts) for p in batch_posts)
            current_before_ts = int(oldest_ts) - 1

            # If we aren't finding new posts anymore, stop early.
            if new_posts == 0:
                break

            time.sleep(sleep_seconds)

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            consecutive_failures += 1
            print(
                f"  -> Network error for week {week_num} (attempt {i + 1}/{max_requests_per_week}): {e}"
            )
            if consecutive_failures >= 5:
                print(f"  -> Too many consecutive network failures for week {week_num}; skipping week.")
                break
            time.sleep(min(60, sleep_seconds * (2 ** min(consecutive_failures, 5))))
            continue
        except ValueError as e:
            consecutive_failures += 1
            print(f"  -> JSON parse error for week {week_num}: {e}")
            time.sleep(min(30, sleep_seconds * (2 ** min(consecutive_failures, 4))))
            continue
        except requests.exceptions.RequestException as e:
            consecutive_failures += 1
            print(f"  -> Request error for week {week_num}: {e}")
            if consecutive_failures >= 5:
                print(f"  -> Too many consecutive request failures for week {week_num}; skipping week.")
                break
            time.sleep(min(60, sleep_seconds * (2 ** min(consecutive_failures, 5))))
            continue
    
    return all_collected_posts

def process_and_sort_posts(collected_posts):
    """
    Filters the collected pool of posts by flair and media, then sorts by score.
    """
    
    filtered_data = []
    for post in collected_posts:
        '''is_text_post = is_text_only_post(post)

        if is_text_post:
            # 2. Select and Format Required Fields
            filtered_data.append({
                'title': post.get('title'),
                'body': post.get('selftext'),
                'upvotes': post.get('score'),
                'num_comments': post.get('num_comments'),
                'timestamp_utc': post.get('created_utc'),
                'author': post.get('author')
            })'''
        filtered_data.append(
            {
                "id": post.get("id"),
                "title": post.get("title"),
                "body": post.get("selftext"),
                "upvotes": int(post.get("score") or 0),
                "num_comments": int(post.get("num_comments") or 0),
                "timestamp_utc": post.get("created_utc"),
                "author": post.get("author"),
            }
        )
    # 3. CRITICAL STEP: Sort all filtered posts by 'upvotes' in descending order
    # This guarantees the final selection is the true top posts from the collected pool.
    print(f"  -> Sorting {len(filtered_data)} filtered posts by upvotes...")
    sorted_posts = sorted(filtered_data, key=lambda p: p["upvotes"], reverse=True)
    
    return sorted_posts

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Weekly Reddit extraction via PullPush")
    parser.add_argument("--subreddit", default=DEFAULT_SUBREDDIT)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--start-week", type=int, default=1)
    parser.add_argument("--end-week", type=int, default=0, help="0 means through the last ISO week")
    parser.add_argument("--max-posts-per-week", type=int, default=DEFAULT_MAX_POSTS_PER_WEEK)
    parser.add_argument("--max-requests-per-week", type=int, default=DEFAULT_MAX_REQUESTS_PER_WEEK)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip weeks whose output file already exists",
    )
    args = parser.parse_args()

    subreddit = args.subreddit
    year = args.year

    print(f"Starting weekly extraction for r/{subreddit} in {year}...")

    max_weeks = datetime.date(year, 12, 28).isocalendar()[1]
    start_week = max(1, args.start_week)
    end_week = max_weeks if args.end_week in (0, None) else min(args.end_week, max_weeks)

    output_dir = args.output_dir or f"weekly_reddit_data_{year}"
    os.makedirs(output_dir, exist_ok=True)

    session = create_session(args.retries)
    total_posts_saved = 0

    for week_num in range(start_week, end_week + 1):
        week_start_ts, week_end_ts = get_week_timestamps(year, week_num)

        if args.resume:
            existing = glob.glob(os.path.join(output_dir, f"{subreddit}_W{week_num}_{year}_top*.json"))
            if existing:
                print(
                    f"  -> Week {week_num}: Skipping (already exists): {os.path.basename(existing[0])}"
                )
                continue

        all_collected_posts = fetch_weekly_data(
            session=session,
            subreddit=subreddit,
            week_num=week_num,
            week_start_ts=week_start_ts,
            week_end_ts=week_end_ts,
            max_posts_per_week=args.max_posts_per_week,
            max_requests_per_week=args.max_requests_per_week,
            sleep_seconds=args.sleep,
            timeout_seconds=args.timeout,
        )

        if not all_collected_posts:
            print(f"  -> Week {week_num}: No data found.")
            continue

        print(f"  -> Week {week_num}: Collected {len(all_collected_posts)} total posts.")

        # Sort by score locally and keep the top N
        sorted_posts = process_and_sort_posts(all_collected_posts)
        final_top_posts = sorted_posts[: args.max_posts_per_week]

        filename = f"{subreddit}_W{week_num}_{year}_top{len(final_top_posts)}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_top_posts, f, ensure_ascii=False, indent=4)

        total_posts_saved += len(final_top_posts)
        print(f"  -> **Week {week_num} SAVED:** {len(final_top_posts)} posts to {filename}")

    print("\n--- EXTRACTION COMPLETE ---")
    print(f"Weeks processed: {start_week}..{end_week} (max ISO weeks in year: {max_weeks})")
    print(f"Total posts saved across processed weeks: {total_posts_saved}")

if __name__ == "__main__":
    main()
