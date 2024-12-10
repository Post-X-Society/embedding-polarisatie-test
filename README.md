# Embedding Polarisatie Test
Kun je de polarisatie detecteren in tweets? Dit repository bevat een poging om een polarisatie test te ontwikkelen op basis van embeddings. De code laat zien hoe je een Random Forest model kunt trainen met behulp van oa TF-IDF, en dat model vervolgens kunt gebruiken om te voorspellen door wie een nieuwe tweet is geschreven.

## Inspiratie
Dit werk is afgeleid van [Twitter Political Compass Machine: A Natural Language Processing Approach and Analysis](https://towardsdatascience.com/twitter-political-compass-machine-a-nature-language-processing-approach-and-analysis-7e8152033495)
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*vZGpGsQD1XQpzSjO)

# Technical Documentation
We continue in English to provide you with technical documentation.

## Overview
This project implements a Natural Language Processing (NLP) tool designed to analyze political discourse on Twitter, with a specific focus on Dutch politicians. The tool analyzes tweets for various linguistic features, sentiment, and writing patterns to understand communication styles and potential polarization in political discourse.

## Features
- Twitter data collection from specified political accounts
- Text preprocessing and cleaning
- Readability analysis
- Sentiment analysis
- Word cloud generation
- Bag of Words (BoW) analysis
- TF-IDF feature extraction
- Machine learning model for politician prediction

## Requirements
- Python 3.x
- Google Colab (for notebook execution)
- Google Drive (for data storage)
- Twitter API credentials

### Required Python Packages
```
tweepy
pandas
nltk
spacy
textstat
wordcloud
scikit-learn
seaborn
matplotlib
pattern
gspread
```

## Setup
1. Mount Google Drive in Colab
2. Configure Twitter API credentials:
   - Consumer key
   - Consumer secret
   - Access key
   - Access secret
3. Install required Python packages
4. Download required NLTK data and spaCy models:
   ```python
   python -m spacy download nl_core_news_lg
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Main Components

### 1. Data Collection
- Fetches Twitter handles from a Google Spreadsheet
- Collects tweets using Twitter API
- Stores raw tweets in CSV format

### 2. Text Preprocessing
- URL removal
- Special character cleaning
- Tokenization
- Stopword removal
- Lemmatization and stemming
- Text normalization

### 3. Analysis Features
- **Readability Analysis**: Implements Flesch reading ease score
- **Sentiment Analysis**: Uses Pattern library for Dutch sentiment analysis
- **Word Clouds**: Generates visual representations of word frequency
- **Bag of Words**: Creates word frequency matrices
- **TF-IDF**: Implements term frequency-inverse document frequency analysis

### 4. Machine Learning
- Random Forest classifier for politician prediction
- Model training and evaluation
- Prediction functionality for new tweets

## Output Files
The tool generates several output files:
- Raw tweets CSV
- Processed tweets CSV
- Word cloud visualizations
- Readability score charts
- Sentiment analysis visualizations
- Trained model files

## Model Performance
The tool includes classification reports and performance metrics for the trained Random Forest model.

## Usage Example
```python
# Load and preprocess data
tweets_df = pd.read_csv('raw-tweets.csv')
processed_df = preprocess_tweets(tweets_df)

# Analyze sentiment
sentiment_scores = analyze_sentiment(processed_df)

# Generate predictions
predicted_politician = predict_politician(new_tweet)
```

## Project Structure
```
├── data/
│   ├── raw-tweets.csv
│   └── processed-tweets.csv
├── output/
│   ├── wordclouds/
│   ├── sentiment-charts/
│   └── models/
└── notebooks/
    └── embedding-polarisatie-test.ipynb
```

## Notes
- The notebook is designed to run in Google Colab
- Requires valid Twitter API credentials
- Processes Dutch language text
- Includes data persistence to Google Drive

## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under GNU GPL

## Disclaimer
This tool is for research purposes only. Ensure compliance with Twitter's Terms of Service when collecting and analyzing data.
