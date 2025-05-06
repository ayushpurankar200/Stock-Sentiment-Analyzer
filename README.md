# Stock Sentiment Analyzer 📰📈

A Streamlit-based web app that combines real‑time stock price data with news‑headline sentiment analysis to help you visualize how market sentiment may correlate with stock movements.

---

## 🚀 Features

* **Live Stock Data**: Fetches historical stock prices for any ticker using [`yfinance`](https://pypi.org/project/yfinance/).
* **Headlines Scraping**: Gathers news headlines from multiple sources (Yahoo Finance, MarketWatch, Seeking Alpha) filtered by your keywords.
* **Sentiment Analysis**: Computes VADER sentiment scores for each headline, and filters to only include non‑neutral articles.
* **Interactive Charts**: Dual‑axis plot of stock price vs. aggregate daily sentiment, with date formatting and hover markers.
* **Customizable**: Choose any ticker symbol, keyword list, time window (7–60 days), and number of articles (1–50).
* **Exportable Data**: Download the merged price+sentiment dataset as a CSV for further analysis.

---

## 📦 Requirements

* Python 3.8 or higher
* Streamlit
* yfinance
* pandas
* nltk
* requests
* beautifulsoup4
* matplotlib

You can install all dependencies via:

```bash
pip install streamlit yfinance pandas nltk requests beautifulsoup4 matplotlib
```

*Note: VADER lexicon is downloaded automatically on first run.*

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   ```

git clone [https://github.com/yourusername/stock-sentiment-analyzer.git](https://github.com/yourusername/stock-sentiment-analyzer.git)
cd stock-sentiment-analyzer

````

2. **Install dependencies**  
   ```bash
pip install -r requirements.txt
````

(Or manually as shown above.)

3. **Run the app**

   ```bash
   ```

streamlit run app\_streamlit.py

````
   If `streamlit` isn’t on your PATH, use:
   ```bash
python -m streamlit run app_streamlit.py
````

4. **Open your browser**
   The app will launch at `http://localhost:8501/` by default.

---

## 🎮 Usage

* **Ticker symbol**: Enter any valid stock ticker (e.g., `AAPL`, `TSLA`, `MSFT`).
* **History (days)**: Select the number of past days to analyze (default: 30).
* **Keywords**: Comma‑separated terms to filter headlines (e.g., `Tesla,Elon Musk`).
* **Articles to fetch**: Number of non‑neutral headlines to display (1–50).
* Click **Run Analysis** to generate the chart, view tables, and export CSV.

---

## 📝 CSV Export

Each run automatically saves the merged dataset to `{TICKER}_sentiment_data.csv`. You can:

* Open in Excel or Google Sheets
* Feed into further data‑science workflows
* Version and track historic sentiment data

---

## 🔧 Configuration

* **User‑Agents and Timeouts**: Adjust in `get_stock_headlines()` if scraping fails.
* **Pool Size Multiplier**: Increase pool size multiplier in `run_analysis()` to fetch more raw headlines before filtering.
* **Environment Variables**: No API keys required — fully scraper‑based.

---

## 🤝 Contributing

1. Fork the repo
2. Create a new feature branch
3. Commit your changes with clear messages
4. Open a Pull Request

---

## 📄 License

MIT © \[Your Name]

---

> Enjoy exploring the interplay between news sentiment and stock performance!
