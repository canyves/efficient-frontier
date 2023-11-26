# Efficient Frontier Monte Carlo Simulation

## Overview

This project explores portfolio optimization by simulating the Efficient Frontier using a Monte Carlo approach. The Efficient Frontier represents a set of optimal portfolios that offer the highest expected return for a defined level of risk. The simulation is conducted using historical stock data fetched from Yahoo Finance.

## Simulation Details

The simulation is built with Python. The Streamlit framework is used to create an interactive web application that allows users to input stock tickers and generate the Efficient Frontier.


## How Portfolios Could Be Found by Minimizing Functions

Instead of using a Monte Carlo simulation, portfolios on the Efficient Frontier can be found by solving optimization problems that either:

- **Maximize the Sharpe Ratio**:

  The Sharpe Ratio is maximized to find the portfolio that provides the highest excess return per unit of risk:

  ![Sharpe Ratio](https://latex.codecogs.com/svg.latex?\Large&space;\max\left(\frac{R_p-R_f}{\sigma_p}\right))

  where \( R_p \) is the portfolio return, \( R_f \) is the risk-free rate, and \( \sigma_p \) is the portfolio standard deviation.

- **Minimize the Variance** for a given level of expected return:

  The variance (or equivalently, the standard deviation) of the portfolio is minimized to find the portfolio with the lowest risk for a given return:

  ![Minimize Variance](https://latex.codecogs.com/svg.latex?\Large&space;\min(\sigma_p^2))

  subject to \( \sum_{i=1}^{N}w_i=1 \) and \( R_p=\text{target return} \), where \( w_i \) are the portfolio weights.

These optimization problems can be solved using techniques such as the Sequential Least Squares Programming (SLSQP) algorithm provided by the `scipy.optimize` module.

