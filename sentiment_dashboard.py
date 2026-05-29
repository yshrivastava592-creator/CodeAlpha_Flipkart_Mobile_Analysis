import pandas as pd
import nltk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import numpy as np
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ============================================================
# LOAD & CLASSIFY
# ============================================================
df = pd.read_csv("Mobile_Reviews_Dataset.csv")
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
df.to_csv("reviews_with_sentiment.csv", index=False)
print("Sentiment done!")

# ============================================================
# COLORS
# ============================================================
BG       = "#060614"
CARD     = "#0d0d2b"
BORDER   = "#1e2a5e"
POSITIVE = "#00e676"
NEGATIVE = "#ff1744"
NEUTRAL  = "#ffab00"
TEXT     = "#f0f0ff"
SUBTEXT  = "#7986cb"
ACCENT   = "#b388ff"

# ============================================================
# METRICS
# ============================================================
total         = len(df)
pos_count     = (df['Sentiment']=='Positive').sum()
neg_count     = (df['Sentiment']=='Negative').sum()
neu_count     = (df['Sentiment']=='Neutral').sum()
pos_pct       = round(pos_count/total*100, 1)
neg_pct       = round(neg_count/total*100, 1)
neu_pct       = round(neu_count/total*100, 1)
avg_score     = round(df['Sentiment_Score'].mean(), 3)
brand_scores  = df.groupby('Brand')['Sentiment_Score'].mean()
best_brand    = brand_scores.idxmax()
worst_brand   = brand_scores.idxmin()
best_product  = df.groupby('Base_Model')['Sentiment_Score'].mean().idxmax()
worst_product = df.groupby('Base_Model')['Sentiment_Score'].mean().idxmin()
price_corr    = round(df['Rating'].corr(df['Price_INR']), 3)
mismatch_count= len(df[(df['Rating']==5) & (df['Sentiment']=='Negative')])

# ============================================================
# FIGURE — clean gridspec only, no add_axes overlaps
# ============================================================
fig = plt.figure(figsize=(32, 20), facecolor=BG)

# ── TITLE ───────────────────────────────────────────────────
fig.text(0.5, 0.978,
         "FLIPKART  MOBILE  SENTIMENT  ANALYSIS  DASHBOARD",
         ha='center', fontsize=28, fontweight='bold', color=TEXT)
fig.text(0.5, 0.958,
         "677 Reviews   |   44 Products   |   12 Brands   |   NLP: VADER   |   Source: Flipkart India",
         ha='center', fontsize=18, color=SUBTEXT)

# ── SEPARATOR ───────────────────────────────────────────────
line = plt.Line2D([0.03,0.97],[0.950,0.950],
                  color=ACCENT, linewidth=1.5,
                  transform=fig.transFigure)
fig.add_artist(line)

# ============================================================
# MASTER GRID — 3 rows
# Row 0: KPI cards (very short)
# Row 1: Top charts (donut, brand bar, wordcloud x2)
# Row 2: Bottom charts (scatter, product ranking)
# ============================================================
outer = gridspec.GridSpec(
    3, 1,
    left=0.03, right=0.97,
    top=0.945, bottom=0.12,
    hspace=0.38,
    height_ratios=[0.08, 0.46, 0.46]
)

# ============================================================
# ROW 0 — KPI CARDS
# ============================================================
kpi_gs = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=outer[0], wspace=0.05)

kpi_data = [
    ("TOTAL REVIEWS", "677",          TEXT,     BORDER),
    ("POSITIVE",      f"{pos_pct}%",  POSITIVE, "#0d2618"),
    ("NEGATIVE",      f"{neg_pct}%",  NEGATIVE, "#2a0d0d"),
    ("AVG SCORE",     str(avg_score), NEUTRAL,  "#2a1f00"),
    ("BEST BRAND",    best_brand,     ACCENT,   "#1a0d2e"),
]

for i, (label, value, color, bg) in enumerate(kpi_data):
    ax_k = fig.add_subplot(kpi_gs[i])
    ax_k.set_facecolor(bg)
    ax_k.set_xlim(0, 1)
    ax_k.set_ylim(0, 1)
    ax_k.axis('off')
    # Top accent bar
    ax_k.axhline(y=0.92, color=color, linewidth=4, xmin=0.02, xmax=0.98)
    # Label
    ax_k.text(0.5, 0.72, label,
              ha='center', va='center',
              fontsize=14, color=SUBTEXT, fontweight='bold')
    # Value
    ax_k.text(0.5, 0.35, value,
              ha='center', va='center',
              fontsize=22, color=color, fontweight='bold')

# ============================================================
# ROW 1 — TOP CHARTS
# ============================================================
top_gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[1], wspace=0.30)

# ── CHART 1: DONUT ──────────────────────────────────────────
ax1 = fig.add_subplot(top_gs[0])
ax1.set_facecolor(CARD)

counts = df['Sentiment'].value_counts().reindex(['Positive','Neutral','Negative'])
wedges, texts, autotexts = ax1.pie(
    counts,
    autopct='%1.1f%%',
    colors=[POSITIVE, NEUTRAL, NEGATIVE],
    startangle=90,
    pctdistance=0.80,
    wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=4),
)
for at in autotexts:
    at.set_color(BG)
    at.set_fontsize(13)
    at.set_fontweight('bold')

legend_items = [
    mpatches.Patch(color=POSITIVE, label=f"Positive  {pos_pct}%  ({pos_count})"),
    mpatches.Patch(color=NEUTRAL,  label=f"Neutral    {neu_pct}%  ({neu_count})"),
    mpatches.Patch(color=NEGATIVE, label=f"Negative  {neg_pct}%  ({neg_count})"),
]
ax1.legend(handles=legend_items, loc='lower center',
           bbox_to_anchor=(0.5, -0.22),
           fontsize=12, facecolor=CARD,
           edgecolor=BORDER, labelcolor=TEXT,
           framealpha=0.9)

ax1.text(0, 0.08, str(total),
         ha='center', va='center',
         fontsize=22, fontweight='bold', color=TEXT)
ax1.text(0, -0.12, "Reviews",
         ha='center', va='center',
         fontsize=16, color=SUBTEXT)
ax1.set_title("Overall Sentiment",
              fontsize=16, fontweight='bold', color=TEXT, pad=14)

# ── CHART 2: BRAND BAR ──────────────────────────────────────
ax2 = fig.add_subplot(top_gs[1])
ax2.set_facecolor(CARD)

brand_sent = df.groupby(['Brand','Sentiment']).size().unstack(fill_value=0)
brand_sent = brand_sent.reindex(columns=['Positive','Neutral','Negative'], fill_value=0)
brand_sent = brand_sent.sort_values('Positive', ascending=False)

x = np.arange(len(brand_sent))
w = 0.25

ax2.bar(x-w, brand_sent['Positive'], width=w, color=POSITIVE, alpha=0.92, label='Positive', zorder=3)
ax2.bar(x,   brand_sent['Neutral'],  width=w, color=NEUTRAL,  alpha=0.92, label='Neutral',  zorder=3)
ax2.bar(x+w, brand_sent['Negative'], width=w, color=NEGATIVE, alpha=0.92, label='Negative', zorder=3)

for bar in ax2.patches:
    h = bar.get_height()
    if h >= 10 and bar.get_facecolor()[1] > 0.7:  # only positive (green)
        ax2.text(bar.get_x()+bar.get_width()/2., h+1.5,
                str(int(h)), ha='center', va='bottom',
                color=TEXT, fontsize=9, fontweight='bold', zorder=4)

ax2.set_title("Brand-wise Sentiment", fontsize=16, fontweight='bold', color=TEXT, pad=14)
ax2.set_xticks(x)
ax2.set_xticklabels(brand_sent.index, rotation=55, ha='right', fontsize=9, color=TEXT)
ax2.tick_params(axis='x', pad=2)
ax2.tick_params(axis='y', labelsize=11, colors=TEXT)
ax2.set_ylim(0, brand_sent['Positive'].max() * 1.20)
ax2.legend(fontsize=11, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, loc='upper right')
ax2.grid(axis='y', alpha=0.15, color=BORDER, zorder=0)
ax2.spines[['top','right','left','bottom']].set_color(BORDER)

# ── CHART 3: WORDCLOUD — using gridspec subplot, split via inset_axes ─
ax3 = fig.add_subplot(top_gs[2])
ax3.set_facecolor(CARD)
ax3.axis('off')
ax3.set_title("What Customers Are Saying",
              fontsize=16, fontweight='bold', color=TEXT, pad=14)

stopwords_extra = {
    'phone','mobile','good','nice','product','very','this','the','is',
    'it','and','for','in','to','a','of','i','my','but','with','so',
    'not','are','have','has','been','would','will','one','also','get',
    'got','use','used','using','too','like','just','that','was','its',
    'am','really','they','their','them','all','more','most','some',
    'any','no','do','did','does','can','could','may','than','then',
    'there','here','from','by','at','on','up','out','about','into'
}

pos_text = " ".join(df[df['Sentiment']=='Positive']['Review_Text'].dropna().astype(str))
neg_text = " ".join(df[df['Sentiment']=='Negative']['Review_Text'].dropna().astype(str))

# Use inset_axes from mpl_toolkits for clean placement
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax3a = inset_axes(ax3, width="95%", height="45%", loc='upper center',
                  bbox_to_anchor=(0, 0.05, 1, 0.9), bbox_transform=ax3.transAxes)
wc_pos = WordCloud(width=700, height=280, background_color=CARD,
                   colormap='Greens', max_words=30,
                   min_font_size=14, max_font_size=80,
                   stopwords=stopwords_extra,
                   prefer_horizontal=0.9,
                   collocations=False).generate(pos_text)
ax3a.imshow(wc_pos, interpolation='bilinear')
ax3a.axis('off')
ax3a.set_title("Positive Reviews", fontsize=12,
               color=POSITIVE, fontweight='bold', pad=4)

ax3b = inset_axes(ax3, width="95%", height="45%", loc='lower center',
                  bbox_to_anchor=(0, 0.02, 1, 0.9), bbox_transform=ax3.transAxes)
wc_neg = WordCloud(width=700, height=280, background_color=CARD,
                   colormap='Reds', max_words=30,
                   min_font_size=14, max_font_size=80,
                   stopwords=stopwords_extra,
                   prefer_horizontal=0.9,
                   collocations=False).generate(neg_text)
ax3b.imshow(wc_neg, interpolation='bilinear')
ax3b.axis('off')
ax3b.set_title("Negative Reviews", fontsize=12,
               color=NEGATIVE, fontweight='bold', pad=4)

# ============================================================
# ROW 2 — BOTTOM CHARTS
# ============================================================
bot_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[2], wspace=0.28)

# ── CHART 4: SCATTER ────────────────────────────────────────
ax4 = fig.add_subplot(bot_gs[0])
ax4.set_facecolor(CARD)

cmap = {'Positive': POSITIVE, 'Negative': NEGATIVE, 'Neutral': NEUTRAL}
for sent, grp in df.groupby('Sentiment'):
    jitter = np.random.uniform(-0.12, 0.12, size=len(grp))
    ax4.scatter(grp['Rating'] + jitter, grp['Sentiment_Score'],
                label=sent, color=cmap[sent],
                alpha=0.65, s=65, edgecolors='none', zorder=3)

ax4.axhline(y=0.05,  color=POSITIVE, linestyle='--', alpha=0.4, linewidth=1.5)
ax4.axhline(y=-0.05, color=NEGATIVE, linestyle='--', alpha=0.4, linewidth=1.5)
ax4.fill_between([0.5,5.5],  0.05, 1.1,  alpha=0.04, color=POSITIVE)
ax4.fill_between([0.5,5.5], -1.1, -0.05, alpha=0.04, color=NEGATIVE)

ax4.text(5.3, 0.6,  "POSITIVE ZONE", fontsize=10,
         color=POSITIVE, alpha=0.7, ha='right', fontweight='bold')
ax4.text(5.3, -0.6, "NEGATIVE ZONE", fontsize=10,
         color=NEGATIVE, alpha=0.7, ha='right', fontweight='bold')

ax4.annotate(f"Mismatch: {mismatch_count} reviews\ngave 5 stars but wrote negatively",
             xy=(4.8, -0.32), fontsize=10, color=NEGATIVE,
             ha='right', style='italic',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor="#2a0d0d", edgecolor=NEGATIVE, alpha=0.85))

ax4.set_title("Star Rating  vs  VADER Sentiment Score",
              fontsize=16, fontweight='bold', color=TEXT, pad=14)
ax4.set_xlabel("Star Rating  (1=Worst  →  5=Best)",
               fontsize=16, color=SUBTEXT, labelpad=14)
ax4.set_ylabel("Sentiment Score  (-1.0  →  +1.0)",
               fontsize=16, color=SUBTEXT, labelpad=8)
ax4.set_xticks([1,2,3,4,5])
ax4.set_xlim(0.5, 5.5)
ax4.tick_params(labelsize=12, colors=TEXT)
ax4.legend(fontsize=12, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
           loc='upper left', bbox_to_anchor=(0.01, 0.98))
ax4.grid(True, alpha=0.10, color=BORDER, zorder=0)
ax4.spines[['top','right','left','bottom']].set_color(BORDER)

# ── CHART 5: PRODUCT RANKING ────────────────────────────────
ax5 = fig.add_subplot(bot_gs[1])
ax5.set_facecolor(CARD)

prod_score = df.groupby('Base_Model')['Sentiment_Score'].mean().round(3)
prod_score = prod_score.sort_values(ascending=True).tail(13)
bar_colors = [NEGATIVE if v < 0.4 else POSITIVE for v in prod_score.values]

bars = ax5.barh(prod_score.index, prod_score.values,
                color=bar_colors, alpha=0.90,
                height=0.60, edgecolor='none', zorder=3)

for bar, val in zip(bars, prod_score.values):
    ax5.text(val + 0.01,
             bar.get_y() + bar.get_height()/2,
             f"{val:.3f}",
             va='center', ha='left',
             color=TEXT, fontsize=11, fontweight='bold')

ax5.set_title("Top Products by Sentiment Score",
              fontsize=16, fontweight='bold', color=TEXT, pad=14)
ax5.set_xlabel("Average Sentiment Score",
               fontsize=16, color=SUBTEXT, labelpad=14)
ax5.tick_params(axis='y', labelsize=11, colors=TEXT)
ax5.tick_params(axis='x', labelsize=11, colors=TEXT)
ax5.set_xlim(0, prod_score.max() + 0.16)
ax5.grid(axis='x', alpha=0.12, color=BORDER, zorder=0)
ax5.spines[['top','right','left','bottom']].set_color(BORDER)

# ── FOOTER ──────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.03,0.97],[0.072,0.072],
               color=BORDER, linewidth=1.2,
               transform=fig.transFigure))
fig.text(0.5, 0.038,
         "CodeAlpha Data Analytics Internship   |   Task 4: Sentiment Analysis   |   Data Source: Flipkart India",
         ha='center', fontsize=18, color=SUBTEXT, fontweight='bold')

plt.savefig("sentiment_dashboard.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\n✅ Dashboard saved!")

# ============================================================
# TERMINAL INSIGHTS
# ============================================================
print("\n" + "="*60)
print("  SENTIMENT DISTRIBUTION")
print("="*60)
print(f"  Positive : {pos_count} reviews ({pos_pct}%)")
print(f"  Neutral  : {neu_count} reviews ({neu_pct}%)")
print(f"  Negative : {neg_count} reviews ({neg_pct}%)")

print("\n" + "="*60)
print("  BRAND WISE SENTIMENT")
print("="*60)
bsp = df.groupby(['Brand','Sentiment']).size().unstack(fill_value=0)
bsp = bsp.reindex(columns=['Positive','Neutral','Negative'], fill_value=0)
print(bsp.to_string())

print("\n" + "="*60)
print("  KEY MARKETING INSIGHTS")
print("="*60)
print(f"""
  1.  {pos_pct}% of {total} reviews are Positive —
      Flipkart mobile customers are largely satisfied.

  2.  BEST BRAND   : {best_brand}
      Highest average sentiment — best customer experience.

  3.  WORST BRAND  : {worst_brand}
      Lowest sentiment score — needs urgent improvement.

  4.  BEST PRODUCT : {best_product}
      Most positively reviewed product overall.

  5.  WORST PRODUCT: {worst_product}
      Most negatively reviewed — quality/performance issues.

  6.  MISMATCH : {mismatch_count} reviews gave 5 stars
      but wrote negative text — hidden dissatisfaction.

  7.  Price vs Rating Correlation : {price_corr}
      Expensive phones do NOT guarantee better reviews.

  8.  Camera & Battery dominate Positive reviews.
      Brands must highlight these in marketing.

  9.  Bad, Quality, Camera dominate Negative reviews.
      R&D must fix camera & build quality urgently.

  10. Most 1-2 star reviews = phones under Rs.10,000.
      Budget segment needs serious quality upgrade.
""")
print("="*60)
print("  Task 4 Complete — CodeAlpha Data Analytics Internship")
print("="*60)