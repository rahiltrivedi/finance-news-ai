import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline
import spacy
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Load AI Models
sentiment_pipeline = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

# Shorten title helper
def shorten_title(title, max_words=12):
    words = title.split()
    important_words = [w for w in words if w.lower() not in ["says", "reports", "claims", "announces", "reveals", "on", "in", "at", "of", "to"]]
    return " ".join(important_words[:max_words])

# Scrape news articles
URL = "https://www.moneycontrol.com/news/business/markets/"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
articles = soup.find_all("li", class_="clearfix")

print(f"📰 Found {len(articles)} articles")

data = []

for article in articles:
    try:
        title_tag = article.find("h2")
        summary_tag = article.find("p")

        if not title_tag or not summary_tag:
            continue

        title = title_tag.get_text(strip=True)
        summary = summary_tag.get_text(strip=True)

        # Shorten the title
        short_title = shorten_title(title)

        # Generate AI-powered summary
        ai_summary = summarizer(summary[:1024], max_length=40, min_length=10, do_sample=False)[0]['summary_text']

        # Sentiment Analysis
        sentiment_result = sentiment_pipeline(summary[:512])[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = round(sentiment_result['score'], 4)

        # Named Entity Recognition
        doc = nlp(summary)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "MONEY", "PERSON"]]

        # Add to data
        data.append({
            "Short_Title": short_title,
            "AI_Insight": ai_summary,
            "Entities": ", ".join(entities),
            "Sentiment_Label": sentiment_label,
            "Sentiment_Score": sentiment_score,
            "Scraped_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        print(f"⚠️ Error parsing one article: {e}")

# Save to Desktop
df = pd.DataFrame(data)
user_home = os.path.expanduser("~")
desktop_path = os.path.join(user_home, "Desktop")
filename = f"indian_finance_news_AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
file_path = os.path.join(desktop_path, filename)
df.to_csv(file_path, index=False, encoding="utf-8-sig")

print(f"✅ CSV saved at: {file_path}")

# Save Sentiment Chart
try:
    df['Sentiment_Label'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title("News Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Article Count")
    plt.tight_layout()
    chart_path = os.path.join(desktop_path, "sentiment_chart_ai.png")
    plt.savefig(chart_path)
    print(f"📊 Chart saved at: {chart_path}")
except Exception as e:
    print(f"⚠️ Could not save chart: {e}")
