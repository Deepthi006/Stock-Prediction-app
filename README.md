# Stock-Prediction-app
Advanced Stock Price Predictor 📈
An interactive web application built with Streamlit and Python for forecasting stock prices using advanced machine learning models and technical analysis. This tool allows users to visualize historical data, analyze technical indicators, and get future price predictions for a wide range of global stocks, indices, and cryptocurrencies.

🔗 Live Demo
[Insert Your Streamlit Community Cloud Link Here]

(After deploying your app, replace the text above with your public URL)

📸 Application Preview
✨ Key Features
Interactive Data Visualization: View historical stock data with interactive Candlestick charts.

Comprehensive Technical Analysis: Automatically plots key indicators like Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Relative Strength Index (RSI), and MACD.

Advanced Machine Learning Models: Choose from a selection of powerful regression models to forecast prices, including:

XGBoost (Extreme Gradient Boosting)

Random Forest

Gradient Boosting

Support Vector Regressor (SVR)

Dynamic Forecasting: Select any number of days into the future to predict, with results displayed in both a table and a graph.

Model Performance Evaluation: Assess model accuracy with key metrics like R-squared (R²) and Mean Squared Error (MSE), and visualize predictions against actual historical values.

User-Friendly Interface: A clean sidebar allows for easy selection of tickers, prediction models, and forecast duration.

Broad Ticker Support: Includes a pre-populated list of major global companies, indices, and cryptocurrencies, plus the option to enter any custom ticker symbol from Yahoo Finance.

🛠️ Technologies & Libraries Used
This project is built primarily with Python and leverages the following powerful libraries:

Web Framework: Streamlit

Data Manipulation: Pandas & NumPy

Financial Data: yfinance

Machine Learning: Scikit-learn, XGBoost

Data Visualization: Plotly

🚀 How to Run Locally
To run this application on your own machine, please follow these steps:

1. Prerequisites:

Make sure you have Python 3.8+ installed on your system.

2. Clone the Repository (Optional):

If you are using Git, you can clone the repository:

git clone [your-github-repo-link]
cd [your-repo-name]

3. Install Dependencies:

This project uses a requirements.txt file to manage its dependencies. Open your terminal and run the following command in the project's root directory:

pip install -r requirements.txt

(Note: The requirements.txt file should contain streamlit, pandas, numpy, yfinance, scikit-learn, xgboost, and plotly).

4. Run the Application:

Once the libraries are installed, run the following command in your terminal:

streamlit run stock_predictor.py

Your web browser should automatically open a new tab with the application running.

📖 How to Use the App
Select a Ticker: Use the "Choose a Stock Ticker" dropdown in the sidebar to select from a list of popular stocks or choose "Enter Custom Ticker..." to input your own.

Choose a Model: Select the machine learning model you'd like to use for the prediction (XGBoost is recommended for its performance).

Set Forecast Period: Use the slider to choose how many days into the future you want to predict.

Analyze the Results:

View the historical data and technical indicator charts.

Check the model's performance metrics (R² and MSE).

Use the "Actual vs. Predicted" graph and table to see how well the model performed on past data.

Review the final forecast table and chart for the future price predictions.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
