# YouTube Comment Sentiment Analysis - Fixed Version
# This version fixes the logical flow issue in your original code

import re, urllib.parse, os, csv, time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import emoji, nltk, seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download required NLTK data
print("📦 Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Configuration
os.environ.get("GOOGLE_API_KEY")
RAW_URL = "https://youtu.be/ix9cRaBkVe0?si=YBB7uGyrHvTUIG-l"

def extract_video_id(url: str) -> str:
    """Extract 11-character YouTube video ID from any YouTube URL"""
    # Try query string v=... first
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if 'v' in qs and len(qs['v'][0]) == 11:
        return qs['v'][0]
    
    # Fall back to regex on the whole URL / path
    m = re.search(r'(?:youtu\.be/|v/|embed/|shorts/|watch\?v=)([A-Za-z0-9_-]{11})', url)
    if m:
        return m.group(1)
    
    raise ValueError("No valid video ID found in URL")

# Extract video ID and set up file paths
VIDEO_ID = extract_video_id(RAW_URL)
CSV_FILE = f"comments_{VIDEO_ID}.csv"
SCORED_FILE = f"comments_{VIDEO_ID}_scored.csv"

print(f"🎯 Video ID: {VIDEO_ID}")
print(f"📁 Will save comments to: {CSV_FILE}")

# ================================
# STEP 1: FETCH YOUTUBE COMMENTS
# ================================
def fetch_comments():
    """Fetch comments from YouTube and save to CSV"""
    print("📥 Fetching comments from YouTube...")
    
    # Build YouTube service
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    def comment_page(page_token=None):
        return youtube.commentThreads().list(
            part="snippet",
            videoId=VIDEO_ID,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText"
        ).execute()
    
    # Fetch and save comments
    fields = ["comment_id", "author", "published_at", "like_count", "text"]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
        total = 0
        token = None
        while True:
            try:
                response = comment_page(token)
            except HttpError as e:
                print(f"❌ HTTP {e.resp.status} – {e.error_details}")
                break
            
            for item in response["items"]:
                s = item["snippet"]["topLevelComment"]["snippet"]
                writer.writerow([
                    item["id"], 
                    s["authorDisplayName"], 
                    s["publishedAt"],
                    s["likeCount"], 
                    s["textDisplay"].replace("\n", " ")
                ])
                total += 1
            print(f"↳ fetched {total:>5} comments ...", end="\r")
            
            token = response.get("nextPageToken")
            if not token:
                break
            time.sleep(0.1)
    
    print(f"\n✅ Finished. {total} comments saved to {CSV_FILE}")
    return total

# Check if we need to fetch comments
if os.path.exists(CSV_FILE):
    print(f"✅ Found existing file: {CSV_FILE}")
    print("⏭️  Skipping comment fetching - using existing data...")
else:
    total_comments = fetch_comments()

# ================================
# STEP 2: LOAD AND PROCESS DATA
# ================================
print("\n📊 Loading comments for sentiment analysis...")

# NOW we can safely read the CSV file (after it's been created)
try:
    df = pd.read_csv(CSV_FILE)
    print(f"📈 Loaded {len(df)} comments")
    print("\n🔍 Sample data:")
    print(df.head())
except FileNotFoundError:
    print(f"❌ Error: Could not find {CSV_FILE}")
    print("Please check if the comment fetching step completed successfully.")
    exit(1)

# ================================
# STEP 3: TEXT CLEANING
# ================================
print("\n🧹 Cleaning comment text...")

STOP_WORDS = set(stopwords.words('english'))

def clean_comment(text: str) -> str:
    if pd.isna(text):
        return ""
    # 1) Demojise 🙂
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 2) Lower-case and strip HTML entities
    text = re.sub(r"&\w+;", " ", text.lower())
    # 3) Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # 4) Collapse usernames & timestamps (@user)
    text = re.sub(r"@\w+", " ", text)
    # 5) Remove punctuation / digits
    text = re.sub(r"[^a-z\s]", " ", text)
    # 6) Collapse repeated characters (loooove → loove)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # 7) Tokenise & drop stop-words
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

tqdm.pandas(desc="Cleaning text")
df["clean"] = df["text"].progress_apply(clean_comment)

# ================================
# STEP 4: SENTIMENT ANALYSIS
# ================================
print("\n🔮 Performing sentiment analysis...")

sia = SentimentIntensityAnalyzer()

def classify(compound_score):
    if compound_score > 0.05:
        return "positive"
    elif compound_score < -0.05:
        return "negative"
    else:
        return "neutral"

tqdm.pandas(desc="Analyzing sentiment")
df["compound"] = df["clean"].progress_apply(lambda t: sia.polarity_scores(t)["compound"])
df["sentiment"] = df["compound"].apply(classify)

# ================================
# STEP 5: SAVE RESULTS
# ================================
print("\n💾 Saving results...")

df.to_csv(SCORED_FILE, index=False, encoding="utf-8")
print(f"✅ Saved labelled data → {SCORED_FILE}")

# ================================
# STEP 6: DISPLAY RESULTS
# ================================
print("\n📊 SENTIMENT ANALYSIS RESULTS:")
print("=" * 40)
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)

print(f"\n📈 SUMMARY:")
print(f"Total comments analyzed: {len(df)}")
print(f"Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/len(df)*100:.1f}%)")
print(f"Neutral:  {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/len(df)*100:.1f}%)")
print(f"Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/len(df)*100:.1f}%)")

# ================================
# STEP 7: VISUALIZATIONS
# ================================
print("\n📊 Creating visualizations...")

try:
    # Sentiment distribution bar chart
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    ax = sns.countplot(x="sentiment", data=df,
                       palette={"positive":"green","neutral":"grey","negative":"red"})
    ax.set(title="Sentiment Distribution", xlabel="Sentiment", ylabel="Comment Count")
    plt.tight_layout()
    plt.savefig(f"sentiment_distribution_{VIDEO_ID}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Word clouds
    for label, colour in [("positive","Greens"), ("negative","Reds")]:
        subset = df.loc[df.sentiment==label, "clean"]
        if len(subset) > 0:
            text_blob = " ".join(subset)
            if text_blob.strip():  # Check if there's actual text
                wc = WordCloud(width=800, height=400,
                               colormap=colour, background_color="white",
                               collocations=False).generate(text_blob)
                plt.figure(figsize=(10,5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Top words in {label} comments")
                plt.tight_layout()
                plt.savefig(f"wordcloud_{label}_{VIDEO_ID}.png", dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print(f"⚠️  No words found for {label} sentiment")
        else:
            print(f"⚠️  No {label} comments found")
    
    print("✅ Visualizations saved as PNG files")
    
except Exception as e:
    print(f"⚠️  Visualization error: {e}")
    print("Charts may not display properly in some environments")

print("\n🎉 Analysis complete!")
print(f"📁 Check these files:")
print(f"   • {CSV_FILE} (raw comments)")
print(f"   • {SCORED_FILE} (with sentiment scores)")
print(f"   • sentiment_distribution_{VIDEO_ID}.png")
print(f"   • wordcloud_positive_{VIDEO_ID}.png")
print(f"   • wordcloud_negative_{VIDEO_ID}.png")