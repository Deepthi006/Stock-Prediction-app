# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, timedelta
import plotly.graph_objects as go

# --- App Configuration ---
st.set_page_config(page_title="Advanced Stock Predictor", layout="wide")

# --- Helper Functions for Feature Engineering and Data Loading ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, span1=12, span2=26, window=9):
    exp1 = data['Close'].ewm(span=span1, adjust=False).mean()
    exp2 = data['Close'].ewm(span=span2, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=window, adjust=False).mean()
    return macd, signal

@st.cache_data
def load_data(ticker_symbol, years=5):
    """
    Fetches historical stock data and calculates technical indicators.
    Caches the data to avoid re-downloading on every interaction.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        # Add technical indicators
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['MACD_Signal'] = calculate_macd(data)
    except Exception:
        return None
        
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# --- Main Application ---
st.title("Advanced Stock Price Predictor 📈")
st.markdown("An interactive tool using machine learning to forecast stock prices.")

# --- Ticker Selection ---
popular_tickers = {
    "Select a Ticker": None,
    # US Companies
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN", "Tesla (TSLA)": "TSLA", "NVIDIA (NVDA)": "NVDA",
    "Meta Platforms (META)": "META", "JPMorgan Chase (JPM)": "JPM",
    # Indian Companies
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS", "Tata Consultancy (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS", "Infosys (INFY.NS)": "INFY.NS",
    # European Companies
    "ASML Holding (ASML.AS)": "ASML.AS", "LVMH (MC.PA)": "MC.PA",
    # Crypto & Indices
    "Bitcoin (BTC-USD)": "BTC-USD", "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 Index (^GSPC)": "^GSPC", "NASDAQ Composite (^IXIC)": "^IXIC",
    "Enter Custom Ticker...": "CUSTOM"
}

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ User Controls")
# --- FIX: Added a unique key to the selectbox to prevent duplicate ID errors ---
selected_ticker_name = st.sidebar.selectbox("Choose a Stock Ticker", list(popular_tickers.keys()), key='ticker_select')

if selected_ticker_name == "Enter Custom Ticker...":
    ticker = st.sidebar.text_input("Enter Custom Stock Ticker", "AAPL").upper()
else:
    ticker = popular_tickers[selected_ticker_name]

model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    ("XGBoost", "RandomForest", "GradientBoosting", "SVR"),
    key='model_select'
)
# --- FIX: Added a unique key to the slider to prevent duplicate ID errors ---
forecast_days = st.sidebar.slider('Days to Forecast', 1, 30, 7, key='forecast_days_slider')

# --- Main App Logic ---
if not ticker:
    st.info("Please select a stock ticker from the sidebar to begin.")
else:
    # --- Data Loading and Display ---
    data_load_state = st.info(f"Loading data for {ticker}...")
    stock_data = load_data(ticker)

    if stock_data is None:
        data_load_state.error(f"Could not load data for '{ticker}'. Please check the ticker symbol and your network connection.")
    else:
        data_load_state.success(f"Data for {ticker} loaded successfully!")
        st.subheader(f"Historical Data for {ticker}")
        st.write(stock_data.tail())

        # --- Technical Analysis Visualization ---
        st.subheader("📊 Technical Analysis")
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data['Date'],
                    open=stock_data['Open'], high=stock_data['High'],
                    low=stock_data['Low'], close=stock_data['Close'], name='Market Data'))
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA_50'], name='50-Day SMA', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['EMA_20'], name='20-Day EMA', line=dict(color='purple')))
        # --- FIX: Updated layout syntax for broader plotly version compatibility ---
        fig.update_layout(
            title=f'{ticker} Price with Technical Indicators',
            yaxis_title='Stock Price (USD)',
            xaxis=dict(rangeslider=dict(visible=False)),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Machine Learning for Price Prediction ---
        st.header("🔮 Advanced Price Prediction")
        
        df = stock_data[['Close']].copy()
        df['HL_PCT'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close'] * 100.0
        df['PCT_change'] = (stock_data['Close'] - stock_data['Open']) / stock_data['Open'] * 100.0
        df['RSI'] = stock_data['RSI']
        df['MACD'] = stock_data['MACD']
        df = df[['Close', 'HL_PCT', 'PCT_change', 'RSI', 'MACD']]
        df.fillna(-99999, inplace=True)

        forecast_col = 'Close'
        df['label'] = df[forecast_col].shift(-forecast_days)
        
        X = np.array(df.drop(columns=['label']))
        X = X[:-forecast_days]
        X_forecast_out = X[-forecast_days:]
        
        df.dropna(inplace=True)
        y = np.array(df['label'])
        
        if len(X) < 10: # Check for a minimum amount of data
            st.warning("Not enough data to train the model. Please choose a stock with a longer history.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner(f"Training the {model_name} model..."):
                if model_name == "RandomForest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name == "GradientBoosting":
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_name == "SVR":
                    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
                elif model_name == "XGBoost":
                    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            st.subheader(f"Model Performance ({model_name})")
            y_pred = model.predict(X_test)
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.2f}")
            col2.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.2f} (closer to 1 is better)")

            # --- NEW: Actual vs. Predicted Values Graph ---
            st.subheader("Actual vs. Predicted Values (on Test Data)")
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual Values', line=dict(color='blue')))
            perf_fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Values', line=dict(color='red', dash='dash')))
            perf_fig.update_layout(xaxis_title="Data Point Index", yaxis_title="Stock Price (USD)")
            st.plotly_chart(perf_fig, use_container_width=True)

            # --- NEW: Actual vs. Predicted Values Table ---
            with st.expander("View Actual vs. Predicted Data Table"):
                comparison_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
                comparison_df['Difference'] = comparison_df['Actual Price'] - comparison_df['Predicted Price']
                st.write(comparison_df)

            # --- Forecasting Section ---
            st.subheader(f"Forecast for the next {forecast_days} days")
            forecast_prediction = model.predict(X_forecast_out)
            
            last_date = stock_data['Date'].iloc[-1]
            forecast_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, forecast_days + 1)])
            
            forecast_df = pd.DataFrame({'Date': forecast_dates.strftime('%Y-%m-%d'), 'Predicted Price': forecast_prediction})
            st.write(forecast_df)

            # --- Forecasting Graph ---
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Historical Close'))
            forecast_fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prediction, name='Forecasted Price', line=dict(color='red', width=3)))
            forecast_fig.update_layout(title=f"Price Forecast for {ticker}", yaxis_title="Stock Price (USD)")
            st.plotly_chart(forecast_fig, use_container_width=True)

