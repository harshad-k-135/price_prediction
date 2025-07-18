# Changelog

All notable changes to the Enhanced Stock Price Prediction with AI Sentiment Analysis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-07-18

### ✨ Major Release: AI Sentiment Analysis Integration

### Added
- 🧠 **FinBERT Integration**: Advanced financial sentiment analysis using ProsusAI/finbert model
- 📰 **Real-time News Feed**: Live Yahoo Finance news with clickable titles and full article access
- 🔗 **Interactive News Interface**: Click any news headline to open source article in new tab
- 🎯 **Sentiment-Enhanced Predictions**: ML model now includes sentiment score as 6th feature
- 🌐 **Multiple News Sources**: Integration with Investing.com, Bloomberg, Barrons, Associated Press
- ⚡ **Single-file Architecture**: Complete application consolidated into one file
- � **Auto-dependency Management**: Automatic installation of required packages
- 📊 **Enhanced UI**: Two-column responsive layout with news panel and sentiment indicators

### Fixed
- 🔄 **Yahoo Finance API Integration**: Updated for latest API structure changes (content.title format)
- 📰 **News Parsing**: Proper extraction of titles, summaries, and URLs from nested JSON structure
- 🎨 **UI/UX Improvements**: Better hover effects, link styling, and responsive design
- � **Increased Coverage**: Now displays 8 recent articles instead of 5

### Enhanced
- � **Universal Stock Support**: Predict any stock available on Yahoo Finance (AAPL, GOOGL, TSLA, ^NSEI, etc.)
- � **Modern Interface**: Glass-morphism design with sentiment badges and color-coded indicators
- ⚡ **Performance**: Optimized parameters (2-year data, 50 epochs) for best accuracy
- 🔮 **Real-time Analysis**: Live sentiment processing with FinBERT on actual news content

### Technical Enhancements
- TensorFlow neural network with 6-dimensional input (OHLCV + Sentiment)
- Transformers + PyTorch integration for FinBERT model
- Advanced news API parsing for Yahoo Finance structure changes
- Responsive HTML/CSS with interactive elements
- Auto-browser opening and port management
- Enhanced error handling and recovery

## [1.0.0] - 2025-07-17

### 🚀 Initial Release: NIFTY Price Prediction Dashboard

### Added
- 🤖 Deep neural network model for stock price prediction
- 📱 Responsive web interface with modern design
- 🌐 RESTful API with comprehensive endpoints
- 📊 Interactive data visualizations
- 📈 Historical data fetching from Yahoo Finance
- 🎨 Glass-morphism UI design with smooth animations
- 🔧 Comprehensive error handling and recovery
- 🛡️ CORS support for cross-origin requests

### Technical Features
- TensorFlow neural network implementation
- Flask backend with threading support
- Vanilla JavaScript frontend (no framework dependencies)
- MinMax scaling for data preprocessing
- 3-layer neural network architecture (64→32→1)
- Yahoo Finance API integration
- Base64 image encoding for charts
- Thread-safe result storage

### Dependencies
- Python 3.8+ support
- NumPy, Pandas, TensorFlow, Flask
- Matplotlib, yfinance, scikit-learn
- Complete dependency auto-installation

### Dependencies
- Python 3.8+ support
- NumPy for numerical computing
- Pandas for data manipulation
- TensorFlow for machine learning
- Flask for web framework
- Matplotlib for visualizations
- yfinance for financial data
- scikit-learn for ML utilities

### Documentation
- Comprehensive README.md with setup instructions
- API endpoint documentation
- Troubleshooting guide
- Contributing guidelines
- MIT License
- Changelog tracking

## [Unreleased]

### Planned Features
- 🔒 User authentication system
- 💾 Prediction history storage
- 📧 Email notifications for predictions
- 🔄 Automated daily predictions
- 📱 Mobile app version
- 🔗 Integration with more financial APIs
- 🎯 Multiple stock symbol support
- 📊 Advanced charting options
- ⚙️ Configurable model parameters
- 🌍 Multi-language support
