"""
Congressional Trading Data Scraper and Prediction Model
Analyzes congressional trades from Capitol Trades to predict stock performance
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

class CongressionalTradingScraper:
    """Scrapes congressional trading data from Capitol Trades"""
    
    def __init__(self):
        self.base_url = "https://www.capitoltrades.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def scrape_trades(self, pages=366):
        """
        Scrape congressional trades from multiple pages
        Returns DataFrame with trade data
        """
        all_trades = []
        
        for page in range(1, pages + 1):
            print(f"Scraping page {page}/{pages}...")
            
            try:
                # URL pattern for pagination
                url = f"{self.base_url}/trades?page={page}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    trades = self._parse_trades_page(soup)
                    all_trades.extend(trades)
                    
                    # Rate limiting to be respectful
                    time.sleep(1)
                else:
                    print(f"Failed to fetch page {page}: Status {response.status_code}")
                    
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_trades)
        
        # Clean and process data
        df = self._clean_trade_data(df)
        
        return df
    
    def _parse_trades_page(self, soup):
        """Parse individual trades from a page"""
        trades = []
        
        # Find trade rows (adjust selectors based on actual HTML structure)
        trade_rows = soup.find_all('tr', class_='trade-row')
        
        for row in trade_rows:
            try:
                trade = {
                    'date': row.find('td', class_='trade-date').text.strip(),
                    'politician': row.find('td', class_='politician-name').text.strip(),
                    'ticker': row.find('td', class_='ticker').text.strip(),
                    'asset_name': row.find('td', class_='asset-name').text.strip(),
                    'transaction_type': row.find('td', class_='transaction-type').text.strip(),
                    'amount': row.find('td', class_='amount').text.strip(),
                    'party': row.find('td', class_='party').text.strip() if row.find('td', class_='party') else 'Unknown',
                    'chamber': row.find('td', class_='chamber').text.strip() if row.find('td', class_='chamber') else 'Unknown'
                }
                trades.append(trade)
            except AttributeError:
                # Skip if any required field is missing
                continue
        
        return trades
    
    def _clean_trade_data(self, df):
        """Clean and standardize trade data"""
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Clean amount ranges and convert to numeric
        df['amount_min'] = df['amount'].apply(self._parse_amount_min)
        df['amount_max'] = df['amount'].apply(self._parse_amount_max)
        df['amount_avg'] = (df['amount_min'] + df['amount_max']) / 2
        
        # Standardize transaction types
        df['transaction_type'] = df['transaction_type'].str.lower()
        df['is_purchase'] = df['transaction_type'].str.contains('buy|purchase', case=False, na=False)
        df['is_sale'] = df['transaction_type'].str.contains('sell|sale', case=False, na=False)
        
        # Remove invalid rows
        df = df.dropna(subset=['date', 'ticker', 'amount_avg'])
        
        # Filter to date range (August 2022 onwards)
        df = df[df['date'] >= '2022-08-01']
        
        return df
    
    def _parse_amount_min(self, amount_str):
        """Parse minimum amount from range string"""
        try:
            # Handle formats like "$1,001 - $15,000"
            amount_str = amount_str.replace('$', '').replace(',', '')
            if '-' in amount_str:
                return float(amount_str.split('-')[0].strip())
            return float(amount_str)
        except:
            return 0
    
    def _parse_amount_max(self, amount_str):
        """Parse maximum amount from range string"""
        try:
            amount_str = amount_str.replace('$', '').replace(',', '')
            if '-' in amount_str:
                return float(amount_str.split('-')[1].strip())
            return float(amount_str)
        except:
            return 0


class StockPerformanceAnalyzer:
    """Analyzes stock performance after congressional trades"""
    
    def __init__(self):
        self.cache = {}  # Cache stock data to reduce API calls
        
    def get_stock_performance(self, ticker, trade_date, periods=[7, 30, 90]):
        """
        Calculate stock performance for different periods after trade
        Returns percentage changes
        """
        
        # Check cache first
        cache_key = f"{ticker}_{trade_date.date()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            
            # Get data for the required period
            start_date = trade_date
            end_date = trade_date + timedelta(days=max(periods) + 10)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {f'return_{p}d': np.nan for p in periods}
            
            # Calculate returns for each period
            returns = {}
            base_price = hist['Close'].iloc[0] if len(hist) > 0 else np.nan
            
            for period in periods:
                try:
                    target_date = trade_date + timedelta(days=period)
                    # Find closest available date
                    closest_idx = hist.index.get_indexer([target_date], method='nearest')[0]
                    
                    if closest_idx < len(hist):
                        future_price = hist['Close'].iloc[closest_idx]
                        returns[f'return_{period}d'] = ((future_price - base_price) / base_price) * 100
                    else:
                        returns[f'return_{period}d'] = np.nan
                except:
                    returns[f'return_{period}d'] = np.nan
            
            # Cache the result
            self.cache[cache_key] = returns
            
            return returns
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return {f'return_{p}d': np.nan for p in periods}
    
    def add_performance_metrics(self, trades_df):
        """Add performance metrics to trades DataFrame"""
        
        print("Fetching stock performance data...")
        performance_data = []
        
        for idx, row in trades_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing trade {idx}/{len(trades_df)}...")
            
            perf = self.get_stock_performance(row['ticker'], row['date'])
            performance_data.append(perf)
            
            # Rate limiting for API calls
            time.sleep(0.1)
        
        # Add performance columns to DataFrame
        perf_df = pd.DataFrame(performance_data)
        trades_df = pd.concat([trades_df, perf_df], axis=1)
        
        return trades_df


class TradingPredictionModel:
    """Machine learning model to predict stock performance based on congressional trades"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, df):
        """Prepare features for model training"""
        
        features_df = df.copy()
        
        # Time-based features
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['month'] = features_df['date'].dt.month
        features_df['quarter'] = features_df['date'].dt.quarter
        
        # Politician features
        politician_trade_counts = df.groupby('politician').size()
        features_df['politician_trade_count'] = features_df['politician'].map(politician_trade_counts)
        
        # Ticker features
        ticker_trade_counts = df.groupby('ticker').size()
        features_df['ticker_popularity'] = features_df['ticker'].map(ticker_trade_counts)
        
        # Party-based features
        party_success_rate = df.groupby('party')['return_30d'].mean()
        features_df['party_avg_return'] = features_df['party'].map(party_success_rate)
        
        # Transaction patterns
        features_df['buy_sell_ratio'] = features_df.groupby('ticker')['is_purchase'].transform('mean')
        
        # Volume features
        features_df['log_amount'] = np.log1p(features_df['amount_avg'])
        
        # Encode categorical variables
        categorical_cols = ['party', 'chamber', 'transaction_type']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].fillna('Unknown'))
            else:
                features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col].fillna('Unknown'))
        
        # Select final features
        feature_cols = [
            'is_purchase', 'is_sale', 'log_amount',
            'day_of_week', 'month', 'quarter',
            'politician_trade_count', 'ticker_popularity',
            'party_avg_return', 'buy_sell_ratio',
            'party_encoded', 'chamber_encoded', 'transaction_type_encoded'
        ]
        
        return features_df[feature_cols]
    
    def train_model(self, df, target='return_30d'):
        """Train the prediction model"""
        
        print("Preparing features...")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target])
        
        # Prepare features
        X = self.prepare_features(df_clean)
        y = df_clean[target]
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training model...")
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train both models
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Ensemble predictions (average)
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Evaluate
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        
        # Store the best model
        self.model = rf_model  # or create ensemble class
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(self.feature_importance.head())
        
        return self
    
    def predict_trades(self, current_trades_df):
        """Predict performance for new trades"""
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        X = self.prepare_features(current_trades_df)
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Add predictions to DataFrame
        current_trades_df['predicted_return_30d'] = predictions
        
        return current_trades_df
    
    def get_top_recommendations(self, current_trades_df, top_n=10):
        """Get top N stock recommendations based on predictions"""
        
        # Get predictions
        trades_with_predictions = self.predict_trades(current_trades_df)
        
        # Aggregate by ticker
        ticker_scores = trades_with_predictions.groupby('ticker').agg({
            'predicted_return_30d': 'mean',
            'is_purchase': 'sum',
            'is_sale': 'sum',
            'amount_avg': 'sum',
            'politician': 'count'
        }).rename(columns={'politician': 'trade_count'})
        
        # Calculate net sentiment
        ticker_scores['net_sentiment'] = ticker_scores['is_purchase'] - ticker_scores['is_sale']
        
        # Combine score (weighted average of prediction and sentiment)
        ticker_scores['combined_score'] = (
            ticker_scores['predicted_return_30d'] * 0.7 +
            ticker_scores['net_sentiment'] * 0.3
        )
        
        # Sort and get top recommendations
        recommendations = ticker_scores.sort_values('combined_score', ascending=False).head(top_n)
        
        return recommendations
    
    def save_model(self, filepath='congress_trading_model.pkl'):
        """Save trained model to file"""
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='congress_trading_model.pkl'):
        """Load trained model from file"""
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        
        print(f"Model loaded from {filepath}")
        return self


class TradingBot:
    """Main trading bot that combines scraping, analysis, and predictions"""
    
    def __init__(self):
        self.scraper = CongressionalTradingScraper()
        self.analyzer = StockPerformanceAnalyzer()
        self.model = TradingPredictionModel()
        self.trades_df = None
        
    def initialize(self, scrape_new_data=True):
        """Initialize the bot with data and trained model"""
        
        if scrape_new_data:
            print("Scraping congressional trades...")
            self.trades_df = self.scraper.scrape_trades(pages=60)
            
            print(f"Scraped {len(self.trades_df)} trades")
            
            # Add performance metrics
            self.trades_df = self.analyzer.add_performance_metrics(self.trades_df)
            
            # Save processed data
            self.trades_df.to_csv('congressional_trades_processed.csv', index=False)
            print("Data saved to congressional_trades_processed.csv")
        else:
            # Load existing data
            self.trades_df = pd.read_csv('congressional_trades_processed.csv')
            self.trades_df['date'] = pd.to_datetime(self.trades_df['date'])
        
        return self
    
    def train(self):
        """Train the prediction model"""
        
        print("\nTraining prediction model...")
        self.model.train_model(self.trades_df)
        self.model.save_model()
        
        return self
    
    def get_daily_recommendations(self, days_back=7):
        """Get recommendations based on recent trades"""
        
        # Filter recent trades
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_trades = self.trades_df[self.trades_df['date'] >= cutoff_date]
        
        print(f"\nAnalyzing {len(recent_trades)} trades from the last {days_back} days...")
        
        # Get recommendations
        recommendations = self.model.get_top_recommendations(recent_trades)
        
        return recommendations
    
    def generate_trading_signals(self, recommendations):
        """Generate actionable trading signals"""
        
        signals = []
        
        for ticker, data in recommendations.iterrows():
            signal = {
                'ticker': ticker,
                'action': 'BUY' if data['combined_score'] > 0 else 'SELL',
                'confidence': min(abs(data['combined_score']) / 10 * 100, 100),  # Convert to percentage
                'predicted_return': data['predicted_return_30d'],
                'congressional_sentiment': 'Bullish' if data['net_sentiment'] > 0 else 'Bearish',
                'trade_volume': data['amount_avg'],
                'number_of_trades': int(data['trade_count']),
                'timestamp': datetime.now().isoformat()
            }
            signals.append(signal)
        
        return signals
    
    def export_signals(self, signals, format='json'):
        """Export trading signals to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f'trading_signals_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2)
        else:
            filename = f'trading_signals_{timestamp}.csv'
            pd.DataFrame(signals).to_csv(filename, index=False)
        
        print(f"Signals exported to {filename}")
        
        return filename


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Congressional Trading Analysis Bot")
    print("=" * 60)
    
    # Initialize bot
    bot = TradingBot()
    
    # Option 1: Scrape new data (set to False to use cached data)
    bot.initialize(scrape_new_data=True)
    
    # Train model
    bot.train()
    
    # Get recommendations
    recommendations = bot.get_daily_recommendations(days_back=30)
    
    print("\n" + "=" * 60)
    print("TOP STOCK RECOMMENDATIONS")
    print("=" * 60)
    print(recommendations)
    
    # Generate trading signals
    signals = bot.generate_trading_signals(recommendations)
    
    print("\n" + "=" * 60)
    print("TRADING SIGNALS")
    print("=" * 60)
    
    for signal in signals[:5]:  # Show top 5
        print(f"\n{signal['ticker']}:")
        print(f"  Action: {signal['action']}")
        print(f"  Confidence: {signal['confidence']:.1f}%")
        print(f"  Predicted 30-day return: {signal['predicted_return']:.2f}%")
        print(f"  Congressional sentiment: {signal['congressional_sentiment']}")
        print(f"  Number of trades: {signal['number_of_trades']}")
    
    # Export signals
    bot.export_signals(signals, format='json')
    
    print("\n" + "=" * 60)
    print("Bot initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
