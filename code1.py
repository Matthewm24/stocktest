import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ssl
import nltk
import matplotlib.pyplot as plt
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

API_KEY = "63851be2fb274f90b998da61c4436fc3"

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data['Price Movement'] = data['Close'].diff()
    data['Direction'] = data['Price Movement'].apply(lambda x: 'Up' if x > 0 else 'Down')
    return data

def fetch_news_articles(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f"Error fetching news articles: {response.status_code}")
        return []

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def analyze_sentiment_huggingface(articles):
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", framework="pt")
    today = datetime.now()
    sentiments = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        published_at = article.get('publishedAt', today.strftime('%Y-%m-%dT%H:%M:%S'))
        published_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
        time_weight = 1 / (1 + (today - published_date).days)
        cleaned_text = clean_text(title + " " + description)
        if cleaned_text.strip():
            result = sentiment_pipeline(cleaned_text)[0]
            result['score'] *= time_weight
            sentiments.append(result)
    return sentiments

def compute_confidence(sentiments):
    positive_sentiments = [s['score'] for s in sentiments if s['label'].upper() == 'POSITIVE']
    negative_sentiments = [s['score'] for s in sentiments if s['label'].upper() == 'NEGATIVE']
    total_positive = sum(positive_sentiments)
    total_negative = sum(negative_sentiments)
    total = total_positive + total_negative
    if total == 0:
        return 0, 0
    confidence_up = (total_positive / total) * 100
    confidence_down = (total_negative / total) * 100
    return confidence_up, confidence_down

def compare_sentiment_with_price(stock_data, confidence_up, confidence_down):
    start_price = stock_data['Close'].iloc[0]
    end_price = stock_data['Close'].iloc[-1]
    actual_movement = 'Up' if end_price > start_price else 'Down'
    predicted_movement = 'Up' if confidence_up > confidence_down else 'Down'
    is_prediction_correct = actual_movement == predicted_movement
    return {
        "start_price": start_price,
        "end_price": end_price,
        "actual_movement": actual_movement,
        "predicted_movement": predicted_movement,
        "is_prediction_correct": is_prediction_correct
    }

def evaluate_sentiment_accuracy(results):
    actual_movements = [res['comparison_result']['actual_movement'] for res in results]
    predicted_movements = [res['comparison_result']['predicted_movement'] for res in results]
    actual_binary = [1 if movement == "Up" else 0 for movement in actual_movements]
    predicted_binary = [1 if movement == "Up" else 0 for movement in predicted_movements]
    accuracy = accuracy_score(actual_binary, predicted_binary)
    precision = precision_score(actual_binary, predicted_binary)
    recall = recall_score(actual_binary, predicted_binary)
    f1 = f1_score(actual_binary, predicted_binary)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def train_predictive_model(stock_data):
    stock_data['Sentiment'] = stock_data['Price Movement'].apply(lambda x: 1 if x > 0 else 0)
    features = stock_data[['Sentiment']]
    target = stock_data['Direction'].apply(lambda x: 1 if x == 'Up' else 0)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


def main():
    ticker = input('Enter the stock ticker: ')
    start_date = input('Enter the start date (YYYY-MM-DD): ')
    end_date = input('Enter the end date (YYYY-MM-DD): ')

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        print(f"No stock data found for {ticker}.")
        return

    news_articles = fetch_news_articles(ticker)
    sentiments = analyze_sentiment_huggingface(news_articles)
    confidence_up, confidence_down = compute_confidence(sentiments)
    comparison_result = compare_sentiment_with_price(stock_data, confidence_up, confidence_down)

    model, model_accuracy = train_predictive_model(stock_data)

    result = {
        "ticker": ticker,
        "confidence_up": confidence_up,
        "confidence_down": confidence_down,
        "comparison_result": comparison_result,
        "model_accuracy": model_accuracy
    }

    print("Results:")
    print(f"Confidence Up: {confidence_up:.2f}%")
    print(f"Confidence Down: {confidence_down:.2f}%")
    print(f"Start Price: {comparison_result['start_price']}")
    print(f"End Price: {comparison_result['end_price']}")
    print(f"Actual Movement: {comparison_result['actual_movement']}")
    print(f"Predicted Movement: {comparison_result['predicted_movement']}")
    print(f"Prediction Correct: {comparison_result['is_prediction_correct']}")
    print(f"Model Accuracy: {model_accuracy}")

if __name__ == "__main__":
    main()