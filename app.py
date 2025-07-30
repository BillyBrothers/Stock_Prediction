# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy import stats
import yfinance as yf


# --- Custom Modules ---
from preprocessing.data_loading import load_stock_data
from preprocessing.feature_engineering import (
    add_lag_prices,
    add_lagged_returns,
    add_moving_averages,
    compute_rolling_stddev,
    add_technical_indicators,
    add_time_features,
    add_volume_features,
    add_price_features
)
from preprocessing.imputation import (
    trim_nans_by_window,
    impute_features
)
from custom_visualizations.interactive_candlesticks import interactive_candlesticks


# Import models
from Modeling.base_model import run_naive_forecast
from Modeling.ARIMA_model import run_arima_forecast
from Modeling.ARIMAX_model import run_arimax_forecast
from Modeling.xg_boost_model import run_xgb_forecast
from Modeling.lstm_model import run_lstm_forecast

# --- Feature Registry ---
FEATURE_FUNCTIONS = {
    "Moving Averages": add_moving_averages,
    "Rolling Std Dev": compute_rolling_stddev,
    "Technical Indicators": add_technical_indicators,
    "Lag Prices": add_lag_prices,
    "Lagged Returns": add_lagged_returns,
    "Volume Features": add_volume_features,
    "Time Features": add_time_features,
    "Price Structure Features": add_price_features,
}

FEATURE_WINDOWS = {
    "Moving Averages": [5, 7, 10, 21, 28, 35],
    "Rolling Std Dev": [7],
    "Technical Indicators": [7, 9, 12, 14, 21, 26, 28, 35],
    "Lag Prices": [1, 2, 3, 4, 5, 6],
    "Lagged Returns": [1, 2, 3, 4, 5, 6],
    "Volume Features": [7, 14, 21, 28, 35, 9],
    "Time Features": [],
    "Price Structure Features": [],
}


# --- Page Config ---
st.set_page_config(layout="wide")
st.title('📊 Stock Prediction and Analysis App')

# --- Sidebar ---
st.sidebar.title("📈 Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")

period = 'max' 
st.sidebar.write(f"**Period:** {period}")


interval = st.sidebar.selectbox("Select Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "1wk", "1mo", "3mo"], index=8)

apply_features = st.sidebar.checkbox("Add Engineered Features")

selected_blocks = st.sidebar.multiselect(
    "Select Feature Blocks",
    options=list(FEATURE_FUNCTIONS.keys()),
    default=list(FEATURE_FUNCTIONS.keys())
) if apply_features else []

# --- Session State Initialization ---
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = pd.DataFrame()

if 'max_calculated_feature_window' not in st.session_state:
    st.session_state.max_calculated_feature_window = 1

if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# --- Load Data Button ---
if st.sidebar.button("Load Data"):
    st.info(f"📦 Loading data for {ticker}...")

    st.session_state.max_calculated_feature_window = 1
    st.session_state.model_results = {} 

    try:
        # --- Load and validate data ---
        raw_data_output = load_stock_data(ticker, period=period, interval=interval)

        if isinstance(raw_data_output, tuple):
            df = raw_data_output[0]
            if not isinstance(df, pd.DataFrame):
                raise TypeError("The first element of the returned tuple is not a DataFrame.")
        elif isinstance(raw_data_output, pd.DataFrame):
            df = raw_data_output
        else:
            raise TypeError(f"load_stock_data returned unexpected type: {type(raw_data_output)}.")


        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            st.error("❌ Loaded data is missing a valid datetime index or contains no rows. This is often caused by Yahoo Finance API limits.")
            st.info("💡 Try a different period/interval combination — such as '5d / 1m' or '1mo / 5m'.")
            st.stop()

        # 🔍 Additional check: ensure critical columns are numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        non_numeric = [col for col in numeric_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]

        if non_numeric:
            st.error(f"❌ Detected invalid column types in: {non_numeric}. This may be due to API restrictions or malformed data.")
            st.info("💡 Try using a shorter period or a coarser interval to get valid numerical data.")


        # --- Display chart ---
        st.subheader("📊 Stock Price Trend (Close Price)")
        interactive_candlesticks(df, ticker)

        
        st.write("Initial shape:", df.shape)

        # --- Adaptive Window Trimming ---
        n_rows = len(df)
        adaptive_feature_windows = {}
        for feat, windows in FEATURE_WINDOWS.items():
            valid = [w for w in windows if w < n_rows]
            adaptive_feature_windows[feat] = valid
            skipped = [w for w in windows if w >= n_rows]
            if skipped:
                st.warning(f"⚠️ Skipped windows {skipped} for '{feat}' — only {n_rows} rows available.")

        # --- Feature Engineering ---
        if apply_features and selected_blocks:
            st.write("🔧 Applying Engineered Features...")

            all_windows_applied = []

            for feat in selected_blocks:
                

                func = FEATURE_FUNCTIONS.get(feat)
                if not func:
                    st.warning(f"⚠️ '{feat}' not found in function registry.")
                    continue

                valid_windows = adaptive_feature_windows.get(feat, [])
                

                try:
                    if feat == "Moving Averages":
                        df = func(df, windows=valid_windows)
                    elif feat == "Rolling Std Dev":
                        df = func(df, windows=valid_windows, target_col='Close')
                    elif feat == "Lag Prices":
                        df = func(df, lag_periods=valid_windows)
                    elif feat == "Lagged Returns":
                        df = func(df, lags=valid_windows)
                    elif feat == "Volume Features":
                        df = func(df, volume_sma_windows=valid_windows)
                    elif feat == "Technical Indicators":
                        df = func(df, windows=valid_windows)
                    else:
                        df = func(df)

                    
                except Exception as e:
                    st.error(f"❌ Error applying '{feat}': {e}")
                    st.stop() 

                # Type safety
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Function '{feat}' did not return a DataFrame. Got {type(df)}")

                all_windows_applied.extend(valid_windows)

            # --- Store max window ---

            if all_windows_applied:
                st.session_state.max_calculated_feature_window = max(
                    st.session_state.max_calculated_feature_window,
                    max(all_windows_applied)
                )

        st.session_state.df_loaded = df
        st.success("✅ Data loaded and features applied!")
        st.write("Post Feature Engineered Shape:", df.shape)

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.exception(e)


df = st.session_state.df_loaded

if df.empty:
    st.info("Please load data using the sidebar to proceed with analysis and modeling.")
    st.stop() # Stop further execution if df is empty


# --- Check and Replace Infinity Values ---
if 'df' in locals(): 
    infinity_count_before_replace = np.isinf(df).sum().sum()
else:
    infinity_count_before_replace = 0

if infinity_count_before_replace > 0:
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
# Account for the largest window size by trimming rows ---
largest_window_size = st.session_state.max_calculated_feature_window

if largest_window_size > 0 and 'df' in locals():
    # Pass the DataFrame and a list containing the single largest window size
    df = trim_nans_by_window(df, [largest_window_size])
    
#+Use the imputation function ---
if 'df' in locals():
    df = impute_features(df)
    st.success("Imputation Complete!")
    st.write("Post Shape Imputation:", df.shape)

# +Calculate remaining NaNs after imputation function ---
if 'df' in locals():
    remaining_nans_after_imputation = df.isna().sum().sum()
else:
    remaining_nans_after_imputation = 0

# If there are remaining NaNs, drop those rows ---
if 'df' in locals():
    final_df = df.copy()
else: 
    None # Work on a copy for this final step if needed
if final_df is not None and remaining_nans_after_imputation > 0:
    
    rows_before_final_drop = len(final_df)
    final_df.dropna(axis=0, inplace=True)
    rows_after_final_drop = len(final_df)

# --- Final Data Quality Check ---
if final_df is not None:
    final_nan_count = final_df.isna().sum().sum()
    final_infinity_count = np.isinf(final_df).sum().sum()

    st.session_state.df_loaded = final_df.copy() 
    st.success("Data loaded, features engineered, and missing/infinite values handled successfully!")
    st.write("Final shape", df.shape)

st.sidebar.markdown("---") 

# --- Model Selection and Predict Button (Back in Sidebar) ---
st.sidebar.subheader("🤖 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ["Naive Forecast", "ARIMA Model", "ARIMAX Model", "XGBoost Model", "LSTM Model"],
    index=0,
    key="sidebar_model_selector"
)

# Display a button to clear all model results
if st.sidebar.button("Clear All Model Results"):
    st.session_state.model_results = {}
    st.rerun() # Rerun to clear the displayed results

# --- Display all previously run model results ---
st.subheader("📈 Model Comparison & Results:")
if not st.session_state.model_results:
    st.info("No model results to display yet. Run a model using the 'Predict' button in the sidebar.")
else:
    for model_name, results in st.session_state.model_results.items():
        st.markdown(f"---")
        st.write(f"### {model_name} Results:")
        st.dataframe(results['results_df']['Actual_Vs_Predicted'])

        if results['actual_values'] and results['predicted_values']:
            st.write(f"### {model_name} Forecast: Actual vs Predicted (First Point of Each Test Split)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(range(len(results['actual_values'])), results['actual_values'], label='Actual Values', color='blue', alpha=0.7)
            ax.scatter(range(len(results['predicted_values'])), results['predicted_values'], label=f'{model_name} Predictions', color=results['plot_color'], alpha=0.7)
            ax.plot(range(len(results['actual_values'])), results['actual_values'], color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
            ax.plot(range(len(results['predicted_values'])), results['predicted_values'], color=results['plot_color'], linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

            title_suffix = f" (Order: {results['params']})" if model_name in ["ARIMA Model", "ARIMAX Model"] else ""
                           #f" (Best Params: {results['params']})" if model_name == "XGBoost Model" else \
                           #f" (Window Size: {results['params']})" if model_name == "LSTM Model" else ""
            ax.set_title(f'{model_name} Forecast{title_suffix}: Actual vs. Predicted per TimeSeriesSplit Fold')
            ax.set_xlabel('Forecast Fold Index')
            ax.set_ylabel('Stock Price')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info(f"No forecast points to plot from {model_name}. Check data or splits.")
    st.markdown("---") 

# --- Predict Button Logic ---
if st.sidebar.button("Predict", key="sidebar_predict_button"):
    if st.session_state.df_loaded is None or st.session_state.df_loaded.empty:
        st.warning("Please load and preprocess data first before running a model.")
    else:
        st.subheader(f"🚀 Running {selected_model}...")
        try:
            results_df = None
            actual_values = None
            predicted_values = None
            model_params = None 
            plot_color = 'gray' 

            if selected_model == "Naive Forecast":
                results_df, actual_values, predicted_values = run_naive_forecast(st.session_state.df_loaded)
                plot_color = 'red'
                st.success("Naive Forecast completed!")
            elif selected_model == "ARIMA Model":
                st.info("Running ARIMA model. This includes hyperparameter tuning and walk-forward validation, which may take a few minutes depending on data size and CPU.")
                progress_text = "ARIMA model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(10, text="Starting ARIMA hyperparameter tuning (this can be slow)...")
                results_df, actual_values, predicted_values, model_params = run_arima_forecast(st.session_state.df_loaded)
                my_bar.progress(100, text="ARIMA model training complete!")
                plot_color = 'green'
                st.success("ARIMA Model completed!")
            elif selected_model == "ARIMAX Model":
                st.info("Running ARIMAX model. This includes hyperparameter tuning and walk-forward validation, which can be slower than ARIMA depending on data size and CPU.")
                progress_text = "ARIMAX model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(10, text="Starting ARIMAX hyperparameter tuning (this can be slow)...")
                results_df, actual_values, predicted_values, model_params = run_arimax_forecast(st.session_state.df_loaded)
                my_bar.progress(100, text="ARIMAX model training complete!")
                plot_color = 'purple'
                st.success("ARIMAX Model completed!")
            elif selected_model == "XGBoost Model":
                st.info("Running XGBoost model with GridSearchCV for hyperparameter tuning. This process can be slow depending on the data size, number of splits, and parameter grid.")
                progress_text = "XGBoost model training and tuning in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(10, text="Starting XGBoost hyperparameter tuning (this can take a while)...")
                results_df, actual_values, predicted_values, model_params = run_xgb_forecast(st.session_state.df_loaded)
                my_bar.progress(100, text="XGBoost model training and tuning complete!")
                plot_color = 'orange'
                st.success("XGBoost Model completed!")
            elif selected_model == "LSTM Model":
                st.info("Running LSTM model. This involves neural network training and walk-forward validation, which can be computationally intensive and may take significant time.")
                progress_text = "LSTM model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                my_bar.progress(10, text="Starting LSTM training...")
                results_df, actual_values, predicted_values = run_lstm_forecast(st.session_state.df_loaded)
                model_params = 35 
                my_bar.progress(100, text="LSTM model training complete!")
                plot_color = 'teal'
                st.success("LSTM Model completed!")

            # Store results in session state
            st.session_state.model_results[selected_model] = {
                'results_df': results_df,
                'actual_values': actual_values,
                'predicted_values': predicted_values,
                'params': model_params,
                'plot_color': plot_color
            }
            
            st.rerun()

        except Exception as e:
            st.error(f"Error running {selected_model} model: {e}")
            st.exception(e) 
st.sidebar.markdown("---")

