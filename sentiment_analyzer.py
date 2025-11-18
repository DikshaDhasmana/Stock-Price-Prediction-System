"""
sentiment_analyzer.py - Financial News Sentiment Analysis with FinBERT

This module:
1. Fetches financial news from NewsAPI
2. Analyzes sentiment using FinBERT (financial domain-specific BERT)
3. Aggregates daily sentiment scores
4. Integrates sentiment features into stock prediction pipeline

Requirements:
    pip install transformers torch newsapi-python

Get your NewsAPI key from: https://newsapi.org/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from typing import List, Dict, Tuple

# News API
try:
    from newsapi import NewsApiClient
except ImportError:
    print("⚠️  NewsAPI not installed. Run: pip install newsapi-python")

# FinBERT for sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not installed. Run: pip install transformers torch")
    FINBERT_AVAILABLE = False


class FinancialSentimentAnalyzer:
    """
    Analyze sentiment of financial news using FinBERT
    
    FinBERT is specifically trained on financial text and outperforms
    general BERT models for finance-related sentiment analysis
    """
    
    def __init__(self, news_api_key=None, model_name='ProsusAI/finbert'):
        """
        Initialize sentiment analyzer
        
        Args:
            news_api_key: NewsAPI key (get from https://newsapi.org/)
            model_name: HuggingFace model name (default: FinBERT)
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.model_name = model_name
        
        # Initialize NewsAPI client
        if self.news_api_key:
            try:
                self.newsapi = NewsApiClient(api_key=self.news_api_key)
                print("✓ NewsAPI client initialized")
            except Exception as e:
                print(f"⚠️  NewsAPI initialization failed: {e}")
                self.newsapi = None
        else:
            print("⚠️  No NewsAPI key provided. Set NEWS_API_KEY environment variable")
            self.newsapi = None
        
        # Initialize FinBERT model
        self.tokenizer = None
        self.model = None
        self.device = None
        
        if FINBERT_AVAILABLE:
            self._load_finbert_model()
    
    def _load_finbert_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            print(f"\nLoading FinBERT model: {self.model_name}")
            print("(First time will download ~500MB model)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ FinBERT loaded on device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load FinBERT: {e}")
            print("   Falling back to rule-based sentiment (less accurate)")
    
    def fetch_news(self, symbol: str, start_date: str, end_date: str, 
                   max_articles: int = 100) -> List[Dict]:
        """
        Fetch financial news from NewsAPI
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_articles: Maximum articles to fetch
            
        Returns:
            List of news articles with metadata
        """
        if not self.newsapi:
            print("❌ NewsAPI not initialized. Cannot fetch news.")
            return []
        
        try:
            # Get company name for better search results
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'META': 'Meta',
                'NVDA': 'Nvidia'
            }
            
            query = f"{company_names.get(symbol, symbol)} stock"
            
            print(f"\nFetching news for {symbol} ({start_date} to {end_date})")
            
            # Fetch news
            response = self.newsapi.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='relevancy',
                page_size=min(max_articles, 100)
            )
            
            articles = response.get('articles', [])
            print(f"✓ Fetched {len(articles)} articles")
            
            return articles
            
        except Exception as e:
            print(f"❌ News fetch failed: {e}")
            return []
    
    def analyze_sentiment_finbert(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using FinBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment_label, confidence_score)
            sentiment_label: 'positive', 'negative', or 'neutral'
            confidence_score: float between 0 and 1
        """
        if not self.model or not self.tokenizer:
            return self._rule_based_sentiment(text)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            sentiment_scores = predictions[0].cpu().numpy()
            sentiment_idx = np.argmax(sentiment_scores)
            sentiment_labels = ['negative', 'neutral', 'positive']
            
            sentiment = sentiment_labels[sentiment_idx]
            confidence = float(sentiment_scores[sentiment_idx])
            
            return sentiment, confidence
            
        except Exception as e:
            print(f"⚠️  FinBERT analysis failed: {e}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based sentiment (less accurate)"""
        text_lower = text.lower()
        
        positive_words = ['gain', 'profit', 'growth', 'rise', 'surge', 'jump', 
                         'high', 'strong', 'beat', 'exceed', 'outperform']
        negative_words = ['loss', 'drop', 'fall', 'decline', 'weak', 'miss',
                         'underperform', 'concern', 'risk', 'cut']
        
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            return 'positive', 0.6
        elif neg_count > pos_count:
            return 'negative', 0.6
        else:
            return 'neutral', 0.5
    
    def calculate_daily_sentiment(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Calculate daily aggregated sentiment scores
        
        Args:
            articles: List of news articles
            
        Returns:
            DataFrame with daily sentiment scores
        """
        if not articles:
            return pd.DataFrame()
        
        print("\nAnalyzing sentiment for each article...")
        
        # Analyze each article
        sentiment_data = []
        for i, article in enumerate(articles):
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if not text.strip():
                continue
            
            # Get sentiment
            sentiment, confidence = self.analyze_sentiment_finbert(text)
            
            # Convert to numerical score: negative=-1, neutral=0, positive=1
            sentiment_score = {'negative': -1, 'neutral': 0, 'positive': 1}[sentiment]
            weighted_score = sentiment_score * confidence
            
            # Extract date
            published_at = article.get('publishedAt', '')
            if published_at:
                date = pd.to_datetime(published_at).date()
            else:
                continue
            
            sentiment_data.append({
                'date': date,
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': confidence,
                'weighted_score': weighted_score,
                'title': article.get('title', '')[:100]  # First 100 chars
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Analyzed {i + 1}/{len(articles)} articles...")
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_data)
        
        if df.empty:
            return df
        
        # Aggregate by day
        daily_sentiment = df.groupby('date').agg({
            'weighted_score': 'mean',  # Average sentiment
            'confidence': 'mean',       # Average confidence
            'score': ['count', 'mean']  # Article count and raw sentiment
        }).reset_index()
        
        daily_sentiment.columns = ['Date', 'Sentiment_Score', 'Sentiment_Confidence', 
                                   'Article_Count', 'Raw_Sentiment']
        
        # Convert date to datetime
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        daily_sentiment = daily_sentiment.set_index('Date')
        
        print(f"\n✓ Daily sentiment calculated for {len(daily_sentiment)} days")
        print(f"  Average sentiment: {daily_sentiment['Sentiment_Score'].mean():.3f}")
        print(f"  Average article count: {daily_sentiment['Article_Count'].mean():.1f} per day")
        
        return daily_sentiment
    
    def integrate_with_features(self, features_df: pd.DataFrame, 
                               sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment scores with technical features
        
        Args:
            features_df: DataFrame with technical features
            sentiment_df: DataFrame with daily sentiment scores
            
        Returns:
            Combined DataFrame with both technical and sentiment features
        """
        print("\nIntegrating sentiment with technical features...")
        
        # Ensure both have datetime index
        if not isinstance(features_df.index, pd.DatetimeIndex):
            features_df.index = pd.to_datetime(features_df.index)
        
        # Merge on date
        combined_df = features_df.join(sentiment_df, how='left')
        
        # Forward fill missing sentiment (use previous day's sentiment)
        sentiment_cols = ['Sentiment_Score', 'Sentiment_Confidence', 
                         'Article_Count', 'Raw_Sentiment']
        combined_df[sentiment_cols] = combined_df[sentiment_cols].fillna(method='ffill')
        
        # Fill any remaining NaNs with neutral sentiment
        combined_df['Sentiment_Score'] = combined_df['Sentiment_Score'].fillna(0)
        combined_df['Sentiment_Confidence'] = combined_df['Sentiment_Confidence'].fillna(0.5)
        combined_df['Article_Count'] = combined_df['Article_Count'].fillna(0)
        combined_df['Raw_Sentiment'] = combined_df['Raw_Sentiment'].fillna(0)
        
        print(f"✓ Combined features: {len(combined_df)} samples, {len(combined_df.columns)} features")
        print(f"  Added sentiment features: {sentiment_cols}")
        
        return combined_df
    
    def save_sentiment_data(self, sentiment_df: pd.DataFrame, symbol: str):
        """Save sentiment data for future use"""
        os.makedirs('data/sentiment', exist_ok=True)
        filepath = f'data/sentiment/{symbol}_sentiment.csv'
        sentiment_df.to_csv(filepath)
        print(f"\n✓ Sentiment data saved to {filepath}")


def demo_sentiment_analysis(symbol='AAPL', days_back=30):
    """
    Demonstration of sentiment analysis system
    
    NOTE: Requires NewsAPI key. Get free key from https://newsapi.org/
    """
    print(f"\n{'='*80}")
    print("FINANCIAL SENTIMENT ANALYSIS DEMO")
    print(f"{'='*80}")
    
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    # Fetch news
    articles = analyzer.fetch_news(
        symbol, 
        start_date.isoformat(), 
        end_date.isoformat()
    )
    
    if not articles:
        print("\n⚠️  No articles fetched. Check your NewsAPI key.")
        print("   Get a free key from: https://newsapi.org/")
        return None
    
    # Calculate sentiment
    sentiment_df = analyzer.calculate_daily_sentiment(articles)
    
    if sentiment_df.empty:
        print("❌ No sentiment data generated")
        return None
    
    # Display sample
    print("\nSample Daily Sentiment:")
    print(sentiment_df.head(10))
    
    # Save data
    analyzer.save_sentiment_data(sentiment_df, symbol)
    
    return sentiment_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Financial sentiment analysis')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    parser.add_argument('--api-key', type=str, help='NewsAPI key')
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['NEWS_API_KEY'] = args.api_key
    
    # Run demo
    sentiment_df = demo_sentiment_analysis(args.symbol, args.days)