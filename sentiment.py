import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Load Data ──────────────────────────────────────────
df = pd.read_csv("Mobile_Reviews_Dataset.csv")

print("=" * 50)
print(f"  Total Reviews  : {len(df)}")
print(f"  Total Products : {df['Base_Model'].nunique()}")
print(f"  Total Brands   : {df['Brand'].nunique()}")
print("=" * 50)

# ── Sentiment Classification ───────────────────────────
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text): return "Neutral"
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:    return "Positive"
    elif score <= -0.05: return "Negative"
    else:                return "Neutral"

def get_score(text):
    if pd.isna(text): return 0
    return round(analyzer.polarity_scores(str(text))['compound'], 4)

df['Sentiment']       = df['Review_Text'].apply(get_sentiment)
df['Sentiment_Score'] = df['Review_Text'].apply(get_score)

# ── Save Output ────────────────────────────────────────
df.to_csv("reviews_with_sentiment.csv", index=False)

# ── Results ────────────────────────────────────────────
print("\n✅ File saved: reviews_with_sentiment.csv")
print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())
print("\nBrand wise Sentiment:")
print(df.groupby(['Brand', 'Sentiment']).size().unstack(fill_value=0))
print("\nDone!")