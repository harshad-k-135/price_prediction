#!/usr/bin/env python3
"""
Universal Stock Price Prediction Dashboard
A single-file application that runs both frontend and backend together
Allows users to predict any stock with customizable training duration
"""

import os
import sys
import webbrowser
import threading
import time
from datetime import datetime, timedelta
import json
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import required packages
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from flask import Flask, jsonify, request, render_template_string
    from flask_cors import CORS
    import requests
    from bs4 import BeautifulSoup
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import re
    from urllib.parse import quote
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                          'numpy', 'pandas', 'yfinance', 'tensorflow', 
                          'scikit-learn', 'matplotlib', 'flask', 'flask-cors',
                          'requests', 'beautifulsoup4', 'transformers', 'torch'])
    # Re-import after installation
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from flask import Flask, jsonify, request, render_template_string
    from flask_cors import CORS
    import requests
    from bs4 import BeautifulSoup
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import re
    from urllib.parse import quote
    DEPENDENCIES_AVAILABLE = True

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
latest_results = {
    'status': 'ready',
    'data': None,
    'timestamp': None,
    'error': None
}

# Initialize FinBERT for sentiment analysis
finbert_tokenizer = None
finbert_model = None

def initialize_finbert():
    """Initialize FinBERT model for financial sentiment analysis"""
    global finbert_tokenizer, finbert_model
    try:
        print("Loading FinBERT model for sentiment analysis...")
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("FinBERT model loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to load FinBERT: {e}")
        return False

def get_company_info(ticker):
    """Get company name and sector from ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Technology')
        return company_name, sector
    except:
        return ticker, 'Technology'

def fetch_yahoo_finance_news(ticker, company_name, days=180):
    """Fetch recent news from Yahoo Finance"""
    try:
        print(f"Fetching news for {ticker} ({company_name})...")
        
        # Get news from yfinance
        stock = yf.Ticker(ticker)
        news_data = stock.news
        
        print(f"Found {len(news_data)} total news articles")
        
        recent_news = []
        for i, article in enumerate(news_data[:8]):  # Get top 8 articles
            # Extract data from the new Yahoo Finance API structure
            content = article.get('content', {})
            
            title = (content.get('title') or 
                    f'Financial News for {ticker} #{i+1}')
            
            summary = (content.get('summary') or 
                      content.get('description') or 
                      'Financial news summary not available')
            
            # Get URL from canonicalUrl or clickThroughUrl
            url = ''
            if content.get('canonicalUrl'):
                url = content.get('canonicalUrl', {}).get('url', '')
            elif content.get('clickThroughUrl'):
                url = content.get('clickThroughUrl', {}).get('url', '')
            if not url:
                url = f'https://finance.yahoo.com/quote/{ticker}'
            
            # Get publisher name
            provider = content.get('provider', {})
            source = provider.get('displayName', 'Yahoo Finance')
            
            # Format publication date
            pub_date_str = content.get('pubDate') or content.get('displayTime')
            if pub_date_str:
                try:
                    # Parse ISO format and convert to readable format
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    pub_date = pub_date.strftime('%Y-%m-%d %H:%M')
                except:
                    pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            else:
                pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            print(f"Article {i+1}: {title[:60]}...")
            
            # Add article to recent news
            recent_news.append({
                'title': title,
                'summary': summary,
                'url': url,
                'published': pub_date,
                'source': source
            })
            print(f"  ✓ Added: {title[:50]}...")
        
        print(f"Successfully processed {len(recent_news)} news articles")
        return recent_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_sentiment_finbert(text):
    """Analyze sentiment using FinBERT"""
    global finbert_tokenizer, finbert_model
    
    if finbert_tokenizer is None or finbert_model is None:
        return 0.0  # Neutral if model not available
    
    try:
        # Clean and truncate text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text[:512]  # FinBERT max length
        
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # FinBERT classes: negative, neutral, positive
        negative, neutral, positive = predictions[0].tolist()
        
        # Convert to sentiment score (-1 to 1)
        sentiment_score = positive - negative
        
        return sentiment_score
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0.0

def calculate_news_sentiment(news_list):
    """Calculate overall sentiment from news articles"""
    if not news_list:
        return 0.0, []
    
    sentiments = []
    for article in news_list:
        # Combine title and summary for sentiment analysis
        text = f"{article['title']} {article['summary']}"
        sentiment = analyze_sentiment_finbert(text)
        sentiments.append(sentiment)
        article['sentiment'] = sentiment
    
    overall_sentiment = np.mean(sentiments) if sentiments else 0.0
    return overall_sentiment, news_list

# HTML Template (Embedded Frontend)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Prediction Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header h1 {
            color: white;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 300;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }

        .header .subtitle {
            color: rgba(255, 255, 255, 0.8);
            text-align: center;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .input-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .input-group {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .news-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            height: fit-content;
        }

        .news-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            color: #333;
        }

        .news-item {
            border-bottom: 1px solid #eee;
            padding: 1rem 0;
            transition: all 0.3s ease;
        }

        .news-item:last-child {
            border-bottom: none;
        }

        .news-item:hover {
            background: rgba(0, 123, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin: 0 -1rem;
        }

        .news-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
            line-height: 1.4;
            font-size: 0.95rem;
        }

        .news-title a {
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s ease;
        }

        .news-title a:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }

        .news-summary {
            color: #666;
            font-size: 0.85rem;
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }

        .news-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: #999;
        }

        .sentiment-badge {
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .sentiment-positive {
            background: #e8f5e8;
            color: #2e7d32;
        }

        .sentiment-negative {
            background: #ffebee;
            color: #c62828;
        }

        .sentiment-neutral {
            background: #f5f5f5;
            color: #666;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #555;
        }

        .form-group input, .form-group select {
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4CAF50;
        }

        .predict-button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin: 1rem auto;
            min-width: 200px;
        }

        .predict-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }

        .predict-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.connected { background: #4CAF50; }
        .status-dot.error { background: #f44336; }
        .status-dot.loading { background: #ff9800; }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .result-card:hover {
            transform: translateY(-5px);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2196F3;
            margin: 0.5rem 0;
        }

        .metric-label {
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .change-positive {
            color: #4CAF50;
        }

        .change-negative {
            color: #f44336;
        }

        .graph-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .graph-container h3 {
            margin-bottom: 1rem;
            color: #333;
        }

        .graph-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #f44336;
            margin: 1rem 0;
        }

        .success-message {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 1rem 0;
        }

        .timestamp {
            color: #666;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 1rem;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 3rem;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .container {
                padding: 0 0.5rem;
            }
            
            .results-grid {
                grid-template-columns: 1fr;
            }
            
            .input-group {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-chart-line"></i> Stock Price Prediction</h1>
        <div class="subtitle">AI-Powered Financial Forecasting Dashboard</div>
    </div>

    <div class="container">
        <div class="main-grid">
            <!-- Left Column: Input and Results -->
            <div class="left-column">
                <!-- Input Card -->
                <div class="input-card">
                    <div class="status-indicator">
                        <div class="status-dot connected" id="statusDot"></div>
                        <span id="statusText">System Ready</span>
                    </div>
                    
                    <div class="input-group">
                        <div class="form-group">
                            <label for="ticker"><i class="fas fa-search"></i> Stock Ticker Symbol</label>
                            <input type="text" id="ticker" placeholder="e.g., AAPL, TSLA, GOOGL, ^NSEI" value="AAPL" />
                            <small style="color: #666; margin-top: 0.5rem;">Model will use 2 years of data with sentiment analysis (50 epochs)</small>
                        </div>
                    </div>
                    
                    <button class="predict-button" id="predictButton" onclick="runPrediction()">
                        <i class="fas fa-brain"></i>
                        <span id="buttonText">Get News & Run AI Prediction</span>
                    </button>
                    <div id="messageContainer"></div>
                </div>

                <!-- Results Grid -->
                <div class="results-grid" id="resultsGrid" style="display: none;">
                    <div class="result-card">
                        <div class="metric-label">Stock Symbol</div>
                        <div class="metric-value" id="stockSymbol">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Company</div>
                        <div class="metric-value" id="companyName" style="font-size: 1.2rem;">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Sector</div>
                        <div class="metric-value" id="companySector" style="font-size: 1.2rem;">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">News Sentiment</div>
                        <div class="metric-value" id="newsSentiment">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Last Trading Date</div>
                        <div class="metric-value" id="lastDate">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Last Closing Price</div>
                        <div class="metric-value" id="lastPrice">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Predicted Date</div>
                        <div class="metric-value" id="predictedDate">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Predicted Price</div>
                        <div class="metric-value" id="predictedPrice">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Expected Change</div>
                        <div class="metric-value" id="expectedChange">-</div>
                    </div>
                    <div class="result-card">
                        <div class="metric-label">Change Percentage</div>
                        <div class="metric-value" id="changePercent">-</div>
                    </div>
                </div>

                <!-- Graph Container -->
                <div class="graph-container" id="graphContainer" style="display: none;">
                    <h3><i class="fas fa-chart-area"></i> Price Prediction with Sentiment Analysis</h3>
                    <img id="graphImage" class="graph-image" alt="Prediction Graph" />
                </div>
            </div>

            <!-- Right Column: News -->
            <div class="right-column">
                <div class="news-card">
                    <div class="news-header">
                        <i class="fas fa-newspaper"></i>
                        <h3>Recent News & Sentiment</h3>
                    </div>
                    <div id="newsContainer">
                        <p style="color: #666; text-align: center; padding: 2rem;">
                            <i class="fas fa-info-circle"></i><br>
                            Enter a ticker symbol and click "Get News & Run AI Prediction" to see recent financial news with sentiment analysis.
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="timestamp" id="lastUpdated"></div>
    </div>

    <div class="footer">
        <p><i class="fas fa-code"></i> Stock Price Prediction Dashboard | Powered by Machine Learning</p>
    </div>

    <script>
        let isRunning = false;

        async function runPrediction() {
            if (isRunning) return;
            
            isRunning = true;
            updateButton(true);
            clearMessages();
            updateStatus('loading', 'Fetching news and processing prediction...');
            
            const ticker = document.getElementById('ticker').value.trim().toUpperCase();
            
            if (!ticker) {
                showMessage('Please enter a valid ticker symbol', 'error');
                isRunning = false;
                updateButton(false);
                updateStatus('error', 'Invalid input');
                return;
            }
            
            try {
                showMessage('Step 1: Fetching recent news and analyzing sentiment...', 'success');
                
                // First get news
                const newsResponse = await fetch('/news', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        ticker: ticker
                    })
                });
                
                if (newsResponse.ok) {
                    const newsData = await newsResponse.json();
                    displayNews(newsData);
                    showMessage('Step 2: Training AI model with sentiment analysis (this may take a few minutes)...', 'success');
                }
                
                // Then run prediction
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        ticker: ticker
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showMessage('AI prediction completed successfully with sentiment analysis!', 'success');
                    displayResults(data.data);
                    loadGraph();
                    updateStatus('connected', 'Prediction completed');
                    document.getElementById('lastUpdated').innerHTML = 
                        `<i class="fas fa-clock"></i> Last updated: ${new Date().toLocaleString()}`;
                } else {
                    showMessage(`Error: ${data.error}`, 'error');
                    updateStatus('error', 'Prediction failed');
                }
            } catch (error) {
                showMessage(`Connection error: ${error.message}`, 'error');
                updateStatus('error', 'Connection failed');
            } finally {
                isRunning = false;
                updateButton(false);
            }
        }

        function displayResults(data) {
            document.getElementById('resultsGrid').style.display = 'grid';
            
            document.getElementById('stockSymbol').textContent = data.ticker;
            document.getElementById('companyName').textContent = data.company_name;
            document.getElementById('companySector').textContent = data.sector;
            
            // Display sentiment with color coding
            const sentimentElement = document.getElementById('newsSentiment');
            const sentiment = data.news_sentiment;
            if (sentiment > 0.1) {
                sentimentElement.textContent = 'Positive';
                sentimentElement.className = 'metric-value change-positive';
            } else if (sentiment < -0.1) {
                sentimentElement.textContent = 'Negative';
                sentimentElement.className = 'metric-value change-negative';
            } else {
                sentimentElement.textContent = 'Neutral';
                sentimentElement.className = 'metric-value';
            }
            
            document.getElementById('lastDate').textContent = data.last_date;
            document.getElementById('lastPrice').textContent = `$${data.last_price.toFixed(2)}`;
            document.getElementById('predictedDate').textContent = data.predicted_date;
            document.getElementById('predictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
            
            const changeElement = document.getElementById('expectedChange');
            const changePercentElement = document.getElementById('changePercent');
            
            changeElement.textContent = `$${data.change.toFixed(2)}`;
            changePercentElement.textContent = `${data.change_percent > 0 ? '+' : ''}${data.change_percent.toFixed(2)}%`;
            
            // Color coding for changes
            const changeClass = data.change >= 0 ? 'change-positive' : 'change-negative';
            changeElement.className = `metric-value ${changeClass}`;
            changePercentElement.className = `metric-value ${changeClass}`;
        }

        function displayNews(newsData) {
            const container = document.getElementById('newsContainer');
            
            if (!newsData.news || newsData.news.length === 0) {
                container.innerHTML = '<p style="color: #666; text-align: center; padding: 2rem;">No recent news found for this ticker.</p>';
                return;
            }
            
            let newsHtml = '';
            newsData.news.forEach(article => {
                const sentiment = article.sentiment;
                let sentimentClass = 'sentiment-neutral';
                let sentimentText = 'Neutral';
                
                if (sentiment > 0.1) {
                    sentimentClass = 'sentiment-positive';
                    sentimentText = 'Positive';
                } else if (sentiment < -0.1) {
                    sentimentClass = 'sentiment-negative';
                    sentimentText = 'Negative';
                }
                
                newsHtml += `
                    <div class="news-item">
                        <div class="news-title">
                            <a href="${article.url}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; text-decoration: none; font-weight: 600;">
                                ${article.title}
                            </a>
                        </div>
                        <div class="news-summary">${article.summary}</div>
                        <div class="news-meta">
                            <span>${article.published} • ${article.source}</span>
                            <span class="sentiment-badge ${sentimentClass}">${sentimentText}</span>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = newsHtml;
        }

        async function loadGraph() {
            try {
                const response = await fetch('/graph');
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('graphImage').src = data.image;
                    document.getElementById('graphContainer').style.display = 'block';
                }
            } catch (error) {
                console.error('Failed to load graph:', error);
            }
        }

        function updateStatus(status, message) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = `status-dot ${status}`;
            statusText.textContent = message;
        }

        function updateButton(loading) {
            const button = document.getElementById('predictButton');
            const buttonText = document.getElementById('buttonText');
            
            if (loading) {
                button.disabled = true;
                buttonText.innerHTML = '<div class="loading-spinner"></div> Processing...';
            } else {
                button.disabled = false;
                buttonText.innerHTML = '<i class="fas fa-brain"></i> Get News & Run AI Prediction';
            }
        }

        function showMessage(message, type) {
            const container = document.getElementById('messageContainer');
            const messageClass = type === 'error' ? 'error-message' : 'success-message';
            container.innerHTML = `<div class="${messageClass}">${message}</div>`;
        }

        function clearMessages() {
            document.getElementById('messageContainer').innerHTML = '';
        }
    </script>
</body>
</html>
"""

def fetch_and_prepare_data(ticker, period='2y', sentiment_score=0.0):
    """Fetch stock data and prepare for training with sentiment analysis"""
    print(f"Fetching {ticker} data for {period} period...")
    
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Remove dividends and stock splits if they exist
    if 'Dividends' in data.columns:
        del data['Dividends']
    if 'Stock Splits' in data.columns:
        del data['Stock Splits']
    
    # Add sentiment score as a feature (broadcast to all rows)
    data['Sentiment'] = sentiment_score
    
    data['Tomorrow'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    
    last_date = data.index[-1]
    scaler = MinMaxScaler()
    
    # Include sentiment in the features
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Tomorrow']
    data_scaled = scaler.fit_transform(data[feature_columns].values)
    X = data_scaled[:, :-1]  # All features except 'Tomorrow'
    y = data_scaled[:, -1]   # 'Tomorrow' prices
    
    return data, X, y, scaler, last_date

def create_and_train_model(X, y, epochs=35):
    """Create and train the neural network model"""
    print(f"Creating and training model with {epochs} epochs...")
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
    
    print("Model training completed!")
    return model, history

def make_prediction(model, X, scaler, last_date, data, sentiment_score=0.0):
    """Make prediction for the next day with sentiment analysis"""
    last_features = X[-1].reshape(1, -1)
    
    # Update sentiment in the features for prediction
    last_features[0, -1] = sentiment_score  # Sentiment is the last feature before target
    
    predicted_price = model.predict(last_features, verbose=0)
    
    # Inverse transform to get actual price
    temp_array = np.zeros((1, scaler.n_features_in_))
    temp_array[0, -1] = predicted_price[0, 0]
    predicted_price_actual = scaler.inverse_transform(temp_array)[0, -1]
    
    next_date = last_date + pd.DateOffset(days=1)
    
    return predicted_price_actual, next_date

def create_visualization(data, predicted_price, next_date, ticker):
    """Create and save visualization"""
    plt.figure(figsize=(14, 8))
    plt.style.use('default')
    
    # Plot last 100 days
    recent_data = data.tail(100)
    plt.plot(recent_data.index, recent_data['Close'], 
             label=f'{ticker} Actual Prices (Last 100 days)', 
             linewidth=2, color='#2196F3')
    
    # Plot prediction
    plt.plot([data.index[-1], next_date], 
             [data['Close'].iloc[-1], predicted_price], 
             color='red', marker='o', linestyle='--', 
             linewidth=3, markersize=8, 
             label=f'Predicted Price ({next_date.strftime("%Y-%m-%d")})')
    
    plt.title(f'{ticker} Price Prediction - Actual vs Predicted', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def home():
    """Serve the main dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/news', methods=['POST'])
def get_news():
    """Get recent news with sentiment analysis"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        
        # Get company info
        company_name, sector = get_company_info(ticker)
        
        # Fetch news
        news_list = fetch_yahoo_finance_news(ticker, company_name)
        
        # Analyze sentiment
        overall_sentiment, news_with_sentiment = calculate_news_sentiment(news_list)
        
        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'overall_sentiment': overall_sentiment,
            'news': news_with_sentiment
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"News fetch error: {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Run prediction with sentiment analysis (fixed: 2y data, 50 epochs)"""
    global latest_results
    
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        
        # Fixed parameters
        period = '2y'
        epochs = 50
        
        print(f"Starting prediction for {ticker} with sentiment analysis")
        
        # Get company info
        company_name, sector = get_company_info(ticker)
        
        # Fetch news and calculate sentiment
        news_list = fetch_yahoo_finance_news(ticker, company_name)
        overall_sentiment, _ = calculate_news_sentiment(news_list)
        sentiment_float = float(overall_sentiment)
        
        print(f"Overall news sentiment: {sentiment_float:.3f}")
        
        # Fetch and prepare data with sentiment
        stock_data, X, y, scaler, last_date = fetch_and_prepare_data(ticker, period, sentiment_float)
        
        # Train model
        model, history = create_and_train_model(X, y, epochs)
        
        # Make prediction with sentiment
        predicted_price, next_date = make_prediction(model, X, scaler, last_date, stock_data, sentiment_float)
        
        # Calculate metrics
        last_price = float(stock_data['Close'].iloc[-1])
        change = predicted_price - last_price
        change_percent = (change / last_price) * 100
        
        # Create visualization
        graph_base64 = create_visualization(stock_data, predicted_price, next_date, ticker)
        
        result_data = {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'news_sentiment': sentiment_float,
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_price': last_price,
            'predicted_date': next_date.strftime('%Y-%m-%d'),
            'predicted_price': float(predicted_price),
            'change': change,
            'change_percent': change_percent,
            'training_period': period,
            'epochs_used': epochs,
            'data_points': len(stock_data)
        }
        
        # Store results
        latest_results = {
            'status': 'completed',
            'data': result_data,
            'timestamp': datetime.now().isoformat(),
            'graph': graph_base64,
            'error': None
        }
        
        print(f"Prediction completed for {ticker}")
        print(f"Company: {company_name} ({sector})")
        print(f"News sentiment: {sentiment_float:.3f}")
        print(f"Last price: ${last_price:.2f}")
        print(f"Predicted price: ${predicted_price:.2f}")
        print(f"Expected change: ${change:.2f} ({change_percent:+.2f}%)")
        
        return jsonify({
            'status': 'success',
            'data': result_data,
            'timestamp': latest_results['timestamp']
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        
        latest_results = {
            'status': 'error',
            'data': None,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        }
        
        return jsonify({'error': error_msg}), 500

@app.route('/graph')
def get_graph():
    """Get the latest prediction graph"""
    if latest_results.get('graph'):
        return jsonify({'image': latest_results['graph']})
    else:
        return jsonify({'error': 'No graph available'}), 404

@app.route('/status')
def status():
    """Get API status"""
    return jsonify({
        'status': 'running',
        'dependencies_available': DEPENDENCIES_AVAILABLE,
        'latest_prediction_status': latest_results['status'],
        'latest_prediction_time': latest_results['timestamp']
    })

def open_browser():
    """Open browser after a delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    """Main function to start the application"""
    print("🚀 Starting Enhanced Stock Price Prediction Dashboard...")
    print(f"✅ Dependencies available: {DEPENDENCIES_AVAILABLE}")
    
    # Initialize FinBERT
    print("🧠 Initializing FinBERT for sentiment analysis...")
    finbert_ready = initialize_finbert()
    
    print("\n📊 Enhanced Features:")
    print("  - Universal stock prediction (any Yahoo Finance ticker)")
    print("  - Real-time news sentiment analysis with FinBERT")
    print("  - Fixed optimal parameters: 2 years data, 50 epochs")
    print("  - Yahoo Finance news integration")
    print("  - Company sector analysis")
    print("  - Sentiment-enhanced AI predictions")
    print("  - Single-file deployment")
    
    if finbert_ready:
        print("✅ FinBERT sentiment analysis ready")
    else:
        print("⚠️ FinBERT not available, using neutral sentiment")
    
    print("\n🌐 Starting web server...")
    print("📱 Opening browser in 2 seconds...")
    
    # Start browser in background
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
