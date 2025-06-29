# 📈 AI-Powered Indian Financial News Scraper

This Python project scrapes the latest financial news from India and applies NLP and AI techniques to analyze the content.

## 🚀 Features

- 🔍 Scrapes news from [Moneycontrol](https://moneycontrol.com)
- 🤖 Summarizes news with Hugging Face's BART model
- 🎯 Performs sentiment analysis using transformers
- 🏷️ Extracts key financial entities using SpaCy
- 📊 Saves output to CSV and generates a sentiment chart

## 🧠 Technologies

- Python
- BeautifulSoup, Requests
- Hugging Face Transformers
- SpaCy (NER)
- Matplotlib
- Pandas

## 📦 Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
