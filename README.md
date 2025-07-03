# Stock Price Prediction with Statistical & Machine Learning Models

This project explores and compares the effectiveness of statistical, machine learning and deep learning models for predicting absolute stock price. I employ a multi-model approach, starting with a Naive method to establish a baseline, traditional time-series method (ARIMA),  time-series modeling yet including additional Exogenous features (ARIMAX), ensemble techniques (XGBoost), and finally leveraging advanced deep learning architectures (LSTM Networks).

The primary goal is to predict whether a given stock's closing price at the end of the X_t will be higher or lower than its closing price at the end of X_t-1, using maximum available preceding data at 1-hour intervals.

## Project Overview

Accurate stock price prediction is an elusive yet highly coveted outcome in finance. This project aims to investigate the predictive power of diverse modeling techniques and evaluate their abilities to accurately predict a stock's price.

My analysis will be conducted on a single selection of a Big Tech company (APPLE) to analyze trends and patterns.

## Methodology

### Data Acquisition and Preprocessing

* **Source:** Historical stock data at 1-hour intervals obtained via `yfinance`.
* **Period:** 2 Years of historical data to capture intra-day volatility.
* **Interval:** 60 Minutes
* **Normalization:** Models that require scaling (XG Boost, LSTM) are transformed using MinMaxScaler to ensure equal contribution to model learning and inverse transformed to original scale for evaluation.
* **Missing Data Handling:** NaN values imputation technique is based upon the characteristics of the particular feature completed post-feature engineering (yFinance imports do not contain NaN values).
  * NaNs dropped from indice 0 to largest window size (35)
  * Forward Fill: Features inherently backward looking and computed from previous data.
    * Rolling Averages, Price lags, Return lags
  * Expanding Median: For statistical constructs 
    * Technical Indicators
  * Linear Interpolation: Capture features that follow gradual trends
      * Price Differences
* **Duplicate Handling**: yfinance provides clean, timestamp-indexed data with no duplicate rows by default. Any accidental duplication introduced during processing is explicitly checked and removed.
* **Outliers Handling**: Outliers were identified using statistical thresholds (IQR & z-score) on key numeric columns but kept in for a complete dataset of historical price trend.

### Feature Engineering
Beyond raw price and volume, the following financial indicators are engineered to provide deeper insights into market trends:
* **Lagged Price:** Previous closing prices shifted by varying time steps to capture temporal dependencies and autocorrelation in price movements.
    * Window Size: Hourly (Within 1 trading day time frame)
* **Lagged Returns:** Historical returns computed from lagged prices using daily, weekly, and monthly time stamps, used to model momentum or mean-reversion behavior in asset prices.
    * Varying Window Size: Hourly, Daily, Weekly
* **Add Moving Average:** (Window Size: Hourly, Daily, Weekly)
  * **Simple Moving Average(**: The unweighted mean of the previous n closing prices using hourly, daily
  * **Exponential Moving Average**: A weighted average that gives more importance to recent prices, making it more responsive to new information.
  * **Logarithmic Moving Average**: A moving average applied to the log-transformed prices to stabilize variance and normalize skewed distributions.
  * **Standard Deviation (Using Target)**: Measures the dispersion of the target variable (e.g., returns) over a rolling window to quantify volatility.
* Technical Indicators (Volatility & Momentum):
  * **Average True Range**: A volatility indicator that measures the average range between high and low prices, accounting for gaps between sessions.
    * Varying Window Size: Hourly, Daily 
  * **Moving Average Convergence Divergence (MACD):** A trend-following momentum indicator calculated as the difference between two EMAs (typically 12 and 26 periods), with a signal line (9-period EMA) to identify crossovers.
    * Window Size: 26, 12, 9 hours
  * **Stochastic Oscillator**: A momentum indicator comparing the current closing price to the high-low range over a set period, used to identify overbought or oversold conditions.
    * Window Size: 14, 3 hours
  * **Average Directional Index**: A non-directional indicator that quantifies the strength of a trend, regardless of its direction, based on the smoothed difference between positive and negative directional movement.
    * Window Size: 14 Hours
  * **Bollinger Bands:** Calculated to characterize A price volatility and identify overbought/oversold conditions. The **width** of the bands is used as a feature.
      * Window Size: 2 Hours
  * **Relative Strength Index (RSI):** A momentum oscillator to measure the speed and change of price movements, identifying potential reversal points.
    * Window Size: 14 Hours
  * **On Balance Volume indicator (OBV):** A cumulative volume-based indicator that adds or subtracts volume based on price direction to detect shifts in buying/selling pressure.
    * Varying Window Size: Hourly, Daily
* Time:
    * Hour
    * Day_of_Week
    * Day_of_Month
    * Month
    * Year
    * Week_of_Year
* **Volume:** 
  * Volume Rolling Average (SMA)
    * Window Size: Hourly, Daily (Within a week time frame)
  * Percentage Change
* **Price Differences:** The price difference between the current and previous High, Low, Open, and Close features.
  * High/Low Range
  * High/Low Range Percentage Change
  * Open/Close Range
  * Open/Close Range Percentage Change

### Feature Importance
Skipped due to Feature Importance evaluation tools producing lower MSE scores. Used all engineered features.

### Walk Forward Approach:
* The time series data was trained on an expanding window of historical data and then tested on the next unseen time steps (training size gradually grew to include prior testing size sequence - testing size stayed constant).
* Fold Size: 100 (Data is divided into 100 folds - each fold contains a training size and testing size.)

### Models:

#### Naïve Model (Baseline)
- Application: Predicts that the next price value will be the same as the most recent observed price. It serves as a baseline model for price prediction tasks, particularly in time series forecasting, where it assumes prices will persist unchanged into the future.
- Advantages: Extremely simple to implement and fast to compute. Despite its simplicity, it can perform well in short-term forecasts or in stable, low-volatility markets. It provides a useful benchmark to ensure that more advanced models are adding value.
- Limitations: Assumes price continuity without considering trends, seasonality, volatility, or external influences. Performs poorly in highly dynamic or mean-reverting environments, and lacks predictive power in long-term forecasts.

#### ARIMA (Autoregressive Integrated Moving Average)
- Application: Used for univariate time series forecasting by capturing temporal structures in historical stock prices. It combines autoregressive (AR) terms, differencing (I) to remove trends and make data stationary, and moving average (MA) components to model past error terms. This model forecasts future price levels based solely on past values and their lagged residual patterns.
- Advantages: Offers transparency, interpretability, and strong performance on datasets with clear linear trends and seasonality (not applied here) after transformation. Useful as a benchmark and when data is limited or lacks complex feature dependencies.
- Limitations: Assumes linearity and stationarity; cannot account for external variables or sudden regime shifts. Its single-variable nature makes it less suitable for capturing broader market dynamics or interactions without extending to models like ARIMAX.
- Parameters used: (1,1,1)

#### ARIMAX (ARIMA with Exogenous Variables)
- Application: Builds upon ARIMA by incorporating external features like the previously engineered ones, improving the model’s ability to account for external influences on price.
- Advantages: Maintains simplicity and interpretability of ARIMA while enhancing predictive power through feature inclusion.
- Limitations: Still linear and sensitive to stationarity assumptions; does not inherently model interactions between features or lagged nonlinear effects
- Parameters used: (1,1,1)

#### XGBoost Regressor (Extreme Gradient Boosting)
- Application: Trained on all engineered features to predict price. Particularly effective at handling non-linear relationships and feature interactions.
- Advantages: High performance, strong regularization, and ability to handle missing or noisy data gracefully.
- Limitations: Requires careful tuning and does not natively model sequential dependencies unless lags are manually included.

#### LSTM (Long Short-Term Memory Networks)
- Application: A type of Recurrent Neural Network (RNN) that is trained on temporal sequences of engineered features. LSTM units learn to retain and forget information through time, making them effective for capturing patterns in time-series data with temporal dependencies.
- Advantages: Excellent at modeling sequences, long-term dependencies, and non-linear dynamics. Adaptable to varying window sizes and architectures (e.g., stacked, bidirectional).
- Limitations: Requires more data and compute than traditional models. May suffer from overfitting without appropriate regularization or dropout.
- Epochs used: 100
- Regularization Techniques used: Dropout, BatchNormalization 

## Evaluation Technique:
The performance of each model (Naive, ARIMA, ARIMAX, XGBoost, LSTM) on the test dataset. Key metrics such as:

* Mean Squared Error (MSE): For numerical prediction tasks.
* Visualizations (e.g., predicted vs. actual price prediction)  to illustrate model behavior and compare their strengths and weaknesses.

## Performance Metric: 
* Mean Squared Error

## Results

* Naive Model: 1.0547
* ARIMA (1,1,1): 0.736
* ARIMAX(1,1,1): 8.7075
* XGBoost Regressor: 0.3701
* LSTM: 44.6802

## Conclusion
On average, XG Boost Regressor most accurately predicted the next hour's stock price using hourly historical data.

#### Investigate ARIMAX & LSTM underperformance:
* ARIMAX
  * Exogenous Features noise
  * Collinearity 
* LSTM 
  * Underfitting due to too few epochs, data scarcity, or insufficient architecture depth
  * Overfitting model memorized noise due to small training sets 
  * Improper Scaling, sequence misalignment, or windowing quirks
* Feature Importance: Discover new methods for evaluating importance

