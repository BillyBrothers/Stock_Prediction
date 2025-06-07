# Stock Price Directional Prediction with Hybrid Machine Learning Models

This project explores and compares the effectiveness of various machine learning and deep learning models for predicting the directional movement of stock prices. I employ a multi-model approach, starting with traditional time-series methods (ARIMA), moving to ensemble techniques (XGBoost), and finally leveraging advanced deep learning architectures (Transformer Networks).

The primary goal is to predict whether a given stock's closing price at the end of the third hour will be higher or lower than its closing price at the end of the second hour, using two hours of preceding data at 5-minute intervals.

## Project Overview

Accurate stock price prediction is an elusive yet highly coveted outcome in finance. This project aims to investigate the predictive power of diverse modeling techniques. By focusing on directional movement rather than exact price prediction, I align with trading strategies where identifying trends is paramount.

My analysis will be conducted on a selection of Big Tech companies (Meta, Apple, Microsoft, Amazon, Alphabet) to analyze trends and patterns.

## Methodology

### Data Acquisition and Preprocessing

* **Source:** Historical stock data at 5-minute intervals obtained via `yfinance`.
* **Period:** 60 days of historical data to capture intra-day volatility.
* **Standardization:** All numerical features are standardized (mean 0, standard deviation 1) to ensure equal contribution to model learning.
* **Missing Data Handling:** Infinite values are replaced with NaNs and rows with NaNs are dropped.

### Feature Engineering

Beyond raw price and volume, the following financial indicators are engineered to provide deeper insights into market trends:

* **Bollinger Bands:** Calculated to characterizeA price volatility and identify overbought/oversold conditions. The **width** of the bands is used as a feature.
* **Relative Strength Index (RSI):** A momentum oscillator to measure the speed and change of price movements, identifying potential reversal points.
* **Rate of Change (ROC):** A momentum oscillator measuring the percentage change between the current price and a price `n` periods ago, useful for detecting divergences.
* **Close Price:** The fundamental closing price.
* **Volume:** Trading volume, indicating liquidity and interest.
* **Price Differences:** The absolute difference between the current and previous closing price.
* **Percentage Change:** The percentage change in closing price from the previous close, highlighting the rate of price changes.

### Model Approaches

I implement and evaluate three distinct modeling approaches:

#### ARIMA (Autoregressive Integrated Moving Average)

* **Application:** ARIMA will be applied to individual stock price series to predict their future values, which will then be used to infer directional movement. This approach serves as a baseline for comparison with more complex models.
* **Limitations:** Primarily designed for linear relationships and may struggle with the inherent non-linearity and volatility of stock markets.

#### XGBoost (Extreme Gradient Boosting)
* **Application:** XGBoost will be trained on the engineered features to classify the directional movement (up or down). It can capture complex interactions between features and is robust to noisy data.
* **Advantages:** Handles non-linearities

#### Transformer Networks

* **Application:** The Transformer model will be trained on sequences of engineered features to directly predict the directional movement. Its parallel processing capabilities and ability to capture long-range dependencies make it particularly suitable for time-series analysis like stock price prediction.
* **Advantages:**: Handles non-linearities



### Results
The performance of each model (ARIMA, XGBoost, Transformer) on the test dataset. Key metrics such as:

* Directional Accuracy: The primary metric, indicating the percentage of times the model correctly predicts the direction of price movement.
* Mean Absolute Error (MAE): For numerical prediction tasks (Transformer, ARIMA).
* F1-Score, Precision, Recall: For classification tasks (XGBoost).
* Visualizations (e.g., predicted vs. actual directional movements, learning curves)  to illustrate model behavior and compare their strengths and weaknesses.
