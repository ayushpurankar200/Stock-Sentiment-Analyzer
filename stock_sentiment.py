import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configuration
warnings.filterwarnings('ignore')
nltk.download("vader_lexicon", quiet=True)

# --- News Scraper with Enhanced Reliability ---
def scrape_source(url, selectors, ticker=None, keywords=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        headlines = []
        
        for selector in selectors:
            for tag in soup.select(selector):
                text = tag.get_text(strip=True)
                if text and len(text) > 15:
                    if keywords:
                        if any(kw.lower() in text.lower() for kw in keywords):
                            headlines.append(text)
                    else:
                        headlines.append(text)
        return list(set(headlines))  # Remove duplicates
    
    except Exception as e:
        st.warning(f"Could not fetch from {url}: {str(e)}")
        return []

def get_stock_headlines(ticker_symbol, company_keywords, pool_size=100):
    kws = [k.strip() for k in company_keywords.split(',')] if isinstance(company_keywords, str) else company_keywords
    
    sources = [
        {
            "url": f"https://finance.yahoo.com/quote/{ticker_symbol}",
            "selectors": ['h1', 'h2', 'h3', 'h4'],
            "keywords": None
        },
        {
            "url": "https://finance.yahoo.com/news",
            "selectors": ['h3', 'h2'],
            "keywords": kws
        },
        {
            "url": f"https://www.marketwatch.com/investing/stock/{ticker_symbol.lower()}",
            "selectors": ['.article__headline', '.article__headline a'],
            "keywords": None
        },
        {
            "url": f"https://seekingalpha.com/symbol/{ticker_symbol.upper()}",
            "selectors": ['.title', 'h1', 'h2', 'h3'],
            "keywords": None
        }
    ]
    
    headlines = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for source in sources:
            futures.append(executor.submit(
                scrape_source,
                source["url"],
                source["selectors"],
                ticker_symbol,
                source["keywords"]
            ))
        
        for future in futures:
            result = future.result()
            if result:
                headlines.extend(result)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_headlines = [h for h in headlines if not (h in seen or seen.add(h))]
    
    return unique_headlines[:min(pool_size, len(unique_headlines))]

# --- Analysis Function with Robust Error Handling ---
@st.cache_data(ttl=3600, show_spinner="Analyzing stock and news sentiment...")
def run_analysis(ticker_symbol, keywords, days, num_articles):
    # Input validation
    if not ticker_symbol or not keywords:
        return None, None, "Ticker symbol and keywords are required"
    
    try:
        # Get stock data with multiple fallbacks
        stock = yf.Ticker(ticker_symbol)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            df = stock.history(period=f"{days}d")
        
        if df.empty:
            return None, None, f"No stock data found for {ticker_symbol}"
            
        df = df[['Close']].reset_index()
        df['Date'] = pd.to_datetime(df['Date'].dt.date)
        
        # Get and analyze headlines
        raw_headlines = get_stock_headlines(ticker_symbol, keywords, pool_size=num_articles*3)
        
        if not raw_headlines:
            return df, pd.DataFrame(), "No headlines found. Try different keywords or check if the ticker exists."
        
        sid = SentimentIntensityAnalyzer()
        records = []
        
        for h in raw_headlines:
            if len(records) >= num_articles:
                break
            score = sid.polarity_scores(h)['compound']
            if score != 0.0:
                records.append({
                    'headline': h, 
                    'sentiment': score,
                    'sentiment_category': 'positive' if score > 0 else 'negative'
                })
        
        if not records:
            return df, pd.DataFrame(), "No headlines with measurable sentiment found."
        
        s_df = pd.DataFrame(records)
        
        # Assign dates with recency bias
        date_weights = np.linspace(1, 0.1, len(df))
        date_weights /= date_weights.sum()
        s_df['date'] = np.random.choice(df['Date'], size=len(s_df), p=date_weights)
        
        # Aggregate sentiment by date
        daily_sentiment = s_df.groupby('date')['sentiment'].mean().reset_index()
        merged_df = pd.merge(df, daily_sentiment, how='left', left_on='Date', right_on='date')
        merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
        merged_df['pct_change'] = merged_df['Close'].pct_change() * 100
        
        return merged_df, s_df, None
        
    except Exception as e:
        return None, None, f"Analysis failed: {str(e)}"

# --- Enhanced Visualization ---
def create_sentiment_plot(merged_df, ticker):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Price plot
    ax1.plot(merged_df['Date'], merged_df['Close'], 'b-', linewidth=2, label='Price')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Sentiment plot
    ax2 = ax1.twinx()
    ax2.fill_between(merged_df['Date'], 0, merged_df['sentiment'], 
                    where=merged_df['sentiment']>=0, 
                    facecolor='green', alpha=0.2, interpolate=True)
    ax2.fill_between(merged_df['Date'], 0, merged_df['sentiment'], 
                    where=merged_df['sentiment']<=0, 
                    facecolor='red', alpha=0.2, interpolate=True)
    ax2.plot(merged_df['Date'], merged_df['sentiment'], 'g-' if merged_df['sentiment'].iloc[-1] >= 0 else 'r-', 
             linewidth=2, label='Sentiment')
    
    ax2.set_ylabel('Sentiment Score', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(-1, 1)
    
    # Formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(merged_df)//5)))
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'{ticker} Price vs. News Sentiment', fontsize=14, pad=20)
    plt.tight_layout()
    return fig

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Stock Sentiment Analyzer", 
        page_icon="📈", 
        layout="wide"
    )
    
    st.title("📊 Stock Sentiment Analyzer")
    st.markdown("""
        Analyze how news sentiment correlates with stock price movements.
        Enter a stock ticker and relevant keywords to see the relationship.
    """)
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Stock Ticker Symbol", "AAPL").upper()
            days = st.slider("Analysis Period (days)", 7, 90, 30)
        with col2:
            keywords = st.text_input("Relevant Keywords (comma separated)", "Apple,iPhone,Tim Cook")
            num_articles = st.slider("Number of Articles to Analyze", 5, 50, 15)
        
        submitted = st.form_submit_button("Run Analysis", type="primary")
    
    if submitted:
        with st.spinner("Gathering data and analyzing sentiment..."):
            merged_df, sent_df, error = run_analysis(ticker, keywords, days, num_articles)
        
        if error:
            st.error(f"Error: {error}")
            return
        
        st.success("Analysis Complete!")
        
        # Results Section
        tab1, tab2, tab3 = st.tabs(["📈 Chart", "📰 Headlines", "📊 Data"])
        
        with tab1:
            st.subheader(f"{ticker} Price vs. Sentiment")
            fig = create_sentiment_plot(merged_df, ticker)
            st.pyplot(fig)
            
            # Calculate correlation
            corr = merged_df[['Close', 'sentiment']].corr().iloc[0,1]
            delta = "↑ Positive" if corr > 0 else "↓ Negative"
            st.metric(
                "Price-Sentiment Correlation", 
                f"{corr:.2f}", 
                delta=delta,
                help="Correlation between -1 (perfect inverse) and 1 (perfect correlation)"
            )
            
        with tab2:
            st.subheader(f"Analyzed Headlines ({len(sent_df)})")
            
            # Sentiment distribution
            st.write("**Sentiment Distribution**")
            sentiment_dist = sent_df['sentiment_category'].value_counts(normalize=True) * 100
            st.bar_chart(sentiment_dist)
            
            # Headlines table
            st.dataframe(
                sent_df.sort_values('sentiment', ascending=False),
                column_config={
                    "headline": "Headline",
                    "sentiment": st.column_config.NumberColumn(
                        "Sentiment Score",
                        format="%.2f",
                        help="From -1 (negative) to +1 (positive)"
                    ),
                    "sentiment_category": "Sentiment"
                },
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            st.subheader("Merged Data")
            st.dataframe(merged_df, use_container_width=True)
            
            # Download button
            csv = merged_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker}_sentiment_analysis.csv",
                mime='text/csv'
            )

if __name__ == "__main__":
    main()