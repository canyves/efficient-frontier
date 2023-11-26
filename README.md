# Efficient Frontier Monte Carlo Simulation

## Overview

This project explores portfolio optimization by simulating the Efficient Frontier using a Monte Carlo approach. The Efficient Frontier represents a set of optimal portfolios that offer the highest expected return for a defined level of risk. The simulation is conducted using historical stock data fetched from Yahoo Finance.

## Simulation Details

The simulation is built with Python. The Streamlit framework is used to create an interactive web application that allows users to input stock tickers and generate the Efficient Frontier.


## How Portfolios Could Be Found by Minimizing Functions

Instead of using a Monte Carlo simulation, portfolios on the Efficient Frontier can be found by solving optimization problems that either:

- **Maximize the Sharpe Ratio**:

  The Sharpe Ratio is maximized to find the portfolio that provides the highest excess return per unit of risk:

  $$\max(SR = \frac{R_p - R_f}{\sigma_p})$$

  where $R_p$ is the portfolio return, $R_f$ is the risk-free rate, and $\sigma_p$ is the portfolio standard deviation.

- **Minimize the Variance** for a given level of expected return:

  The variance (or equivalently, the standard deviation) of the portfolio is minimized to find the portfolio with the lowest risk for a given return:

  $$\min(\sigma_p^2)$$

  subject to $\sum_{i=1}^{N}w_i=1$ and $R_p=\text{target return}$, where $w_i$ are the portfolio weights.


## Launching the Application

To run the Streamlit application, you need to have Streamlit installed, you can do so using pip:

```bash
pip install streamlit
```

Launch the server using:

```bash
streamlit run efficient_frontier.py
```

You can then access the server on localhost:8501, choose stocks to run the simulation on and visualise the efficient frontier.

<img width="434" alt="Screenshot 2023-11-26 at 10 02 05" src="https://github.com/canyves/efficient_frontier/assets/134456846/5e07475b-0fc1-4d4f-bbaa-56e0f64c037d"> 