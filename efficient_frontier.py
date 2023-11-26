import numpy as np
import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from typing import List, Tuple

# Override the default pandas_datareader data fetching method with yfinance's method
yf.pdr_override()

# Define type aliases for clarity
Vector = np.ndarray
Matrix = np.ndarray

def fetch_stock_data(tickers: List[str], start_date: datetime.date, end_date: datetime.date) -> Tuple[Vector, Matrix, pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance and calculate daily returns, mean returns, and the covariance matrix.
    
    Parameters:
    tickers (List[str]): List of stock tickers to fetch data for.
    start_date (datetime.date): The start date for fetching data.
    end_date (datetime.date): The end date for fetching data.
    
    Returns:
    Tuple[Vector, Matrix, pd.DataFrame]: mean returns vector, covariance matrix of returns, and raw daily returns DataFrame.
    """
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    covariance_matrix = returns.cov()
    return mean_returns, covariance_matrix, returns

def portfolio_performance(weights: Vector, mean_returns: Vector, cov_matrix: Matrix) -> Tuple[float, float]:
    """
    Calculate the expected return and standard deviation (volatility) of a portfolio.
    
    Parameters:
    weights (Vector): Asset weights in the portfolio.
    mean_returns (Vector): Mean returns for each asset.
    cov_matrix (Matrix): Covariance matrix of returns.
    
    Returns:
    Tuple[float, float]: Portfolio standard deviation and expected return.
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_std, portfolio_return

# Streamlit UI setup
st.title("Efficient Frontier Monte Carlo Simulation")

# User input for stock tickers
user_tickers = [ticker.strip() for ticker in st.text_input("Enter stock tickers, separated by commas").split(",")]
start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

# Constants for the Monte Carlo simulation
num_portfolios = 25000
risk_free_rate = 0.01  # Risk-free rate for Sharpe ratio calculation

if st.button("Generate Efficient Frontier"):
    # Fetch stock data
    mean_returns, covariance_matrix, returns = fetch_stock_data(user_tickers, start_date, end_date)
    num_assets = len(user_tickers)
    results_array = np.zeros((3 + num_assets, num_portfolios))
    
    # Run Monte Carlo simulation to generate random portfolios
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_std_dev, portfolio_return = portfolio_performance(weights, mean_returns, covariance_matrix)
        results_array[0, i] = portfolio_std_dev
        results_array[1, i] = portfolio_return
        results_array[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe ratio
        results_array[3:, i] = weights  # Portfolio weights
    
    # Create a DataFrame with the simulation results
    results_df = pd.DataFrame(results_array.T, columns=['std_dev', 'returns', 'sharpe'] + user_tickers)
    
    # Identify the portfolios with the maximum Sharpe ratio and minimum standard deviation
    max_sharpe_idx = np.argmax(results_array[2])
    sdp, rp = results_array[0, max_sharpe_idx], results_array[1, max_sharpe_idx]
    max_sharpe_allocation = results_df.iloc[max_sharpe_idx, 3:]

    min_std_idx = np.argmin(results_array[0])
    sdp_min, rp_min = results_array[0, min_std_idx], results_array[1, min_std_idx]
    min_std_allocation = results_df.iloc[min_std_idx, 3:]

    # Plot the Efficient Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results_df.std_dev, results_df.returns, c=results_df.sharpe, cmap='viridis')
    ax.scatter(sdp, rp, color='r', s=150, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min, rp_min, color='b', s=150, label='Minimum volatility')
    ax.set_title("Simulated Portfolio Optimization based on Efficient Frontier")
    ax.set_xlabel('Volatility (Std. Deviation)')
    ax.set_ylabel('Expected Returns')
    ax.legend()

    st.pyplot(fig)

    # Display portfolio details on Streamlit
    st.subheader('Portfolio with Maximum Sharpe Ratio')
    st.json(max_sharpe_allocation.to_json())

    st.subheader('Portfolio with Minimum Variance')
    st.json(min_std_allocation.to_json())
