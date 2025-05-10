import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data['Price Movement'] = data['Close'].diff()
    data['Direction'] = data['Price Movement'].apply(lambda x: 1 if x > 0 else 0)
    return data

def analyze_sentiment(articles):
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", framework="pt")
    today = datetime.now()
    sentiments = []
    for article in articles:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        published_at = article.get('publishedAt', today.strftime('%Y-%m-%dT%H:%M:%SZ'))
        try:
            published_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            published_date = today
        time_weight = 1 / (1 + (today - published_date).days)
        cleaned_text = title + " " + description
        if cleaned_text.strip():
            result = sentiment_pipeline(cleaned_text)[0]
            sentiment_score = result['score'] if result['label'].upper() == 'POSITIVE' else -result['score']
            weighted_score = sentiment_score * time_weight
            sentiments.append(weighted_score)
    return np.mean(sentiments) if sentiments else 0

def compute_sentiment_trends(sentiment_scores, window=5):
    sentiment_df = pd.DataFrame({'Sentiment': sentiment_scores})
    sentiment_df['Sentiment_MA'] = sentiment_df['Sentiment'].rolling(window=window).mean()
    return sentiment_df

def create_historical_features(data):
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['Volatility'] = data['Close'].rolling(window=5).std()
    data = data.dropna()
    return data

def train_predictive_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

def automated_weight_tuning(features, target, param_grid):
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5
    )
    grid_search.fit(features, target)
    return grid_search.best_params_, grid_search.best_score_

def main():
    ticker = input('Enter the stock ticker: ')
    start_date = input('Enter the start date (YYYY-MM-DD): ')
    end_date = input('Enter the end date (YYYY-MM-DD): ')

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        print("No stock data found.")
        return

    news_articles = [
        {'title': 'Stock surges after earnings', 'description': 'Company beats expectations', 'publishedAt': '2024-01-01T12:00:00Z'},
        {'title': 'Market declines on inflation fears', 'description': 'Stocks fall broadly', 'publishedAt': '2024-01-02T12:00:00Z'}
    ]
    sentiment_score = analyze_sentiment(news_articles)
    print(f"Weighted Sentiment Score: {sentiment_score:.2f}")

    historical_features = create_historical_features(stock_data)
    historical_features['Sentiment'] = sentiment_score


    features = historical_features[['Sentiment', 'SMA_5', 'SMA_10', 'Volatility']]
    target = historical_features['Direction']


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)


    param_grid = {
        'max_depth': [3, 5, 7], 
        'n_estimators': [50, 100, 150]
    }

    best_params, best_score = automated_weight_tuning(scaled_features, target, param_grid)
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_score:.2f}")

    model, accuracy = train_predictive_model(scaled_features, target)
    print(f"Final Model Accuracy: {accuracy:.2f}")


    recent_features = scaled_features[-1].reshape(1, -1)
    predicted_direction = model.predict(recent_features)
    print(f"The stock price is predicted to go: {'Up' if predicted_direction[0] == 1 else 'Down'}")

if __name__ == "__main__":
    main()