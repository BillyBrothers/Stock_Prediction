import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Import models
from Modeling.base_model import run_naive_forecast
from Modeling.ARIMA_model import run_arima_forecast
from Modeling.ARIMAX_model import run_arimax_forecast
from Modeling.xg_boost_model import run_xgb_forecast
from Modeling.lstm_model import run_lstm_forecast # Import LSTM model

# Map features to functions
FEATURE_FUNCTIONS = {
    "Lag Prices": add_lag_prices,
    "Lagged Returns": add_lagged_returns,
    "Moving Averages": add_moving_averages,
    "Rolling Std Dev": compute_rolling_stddev,
    "Technical Indicators": add_technical_indicators,
    "Time Features": add_time_features,
    "Volume Features": add_volume_features,
    "Price Structure Features": add_price_features
}

st.set_page_config(layout="wide")
st.title('Stock Prediction and Analysis App')

# --- Sidebar layout ---
st.sidebar.title("📈 Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=3)
interval = st.sidebar.selectbox("Select Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo", "3mo"], index=8)


apply_features = st.sidebar.checkbox("Add Engineered Features")

if apply_features:
    selected_blocks = st.sidebar.multiselect(
        "Select Feature Blocks",
        options=list(FEATURE_FUNCTIONS.keys()),
        default=["Lag Prices", "Lagged Returns", "Moving Averages", "Rolling Std Dev", "Technical Indicators", "Time Features", "Volume Features", "Price Structure Features"]
    )
else:
    selected_blocks = []

# --- Data Loading and Preprocessing Logic (triggered by Load Data button) ---
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'max_calculated_feature_window' not in st.session_state:
    st.session_state.max_calculated_feature_window = 1


if st.sidebar.button("Load Data"):
    st.info(f"Loading data for {ticker}...")

    # Reset max_calculated_feature_window on new data load
    st.session_state.max_calculated_feature_window = 1

    try:
        raw_data_output = load_stock_data(ticker, period=period, interval=interval)

        if isinstance(raw_data_output, tuple):
            df = raw_data_output[0]
            if not isinstance(df, pd.DataFrame):
                raise TypeError("The first element of the tuple is not a pandas DataFrame.")
        elif isinstance(raw_data_output, pd.DataFrame):
            df = raw_data_output
        else:
            raise TypeError(f"load_stock_data returned an unexpected type: {type(raw_data_output)}. Expected pandas DataFrame or a tuple containing one.")

        st.subheader("📊 Stock Price Trend (Close Price)")
        st.line_chart(df['Close'])

        # --- Initial NaN Check (Before Feature Engineering) ---
        initial_nan_count = df.isna().sum().sum()
        st.write(f"Initial NaN count (before feature engineering): **{initial_nan_count}**")

        if apply_features and selected_blocks:
            st.write("🔧 Applying Engineered Features...")

            all_windows_applied = []

            for feat in selected_blocks:
                if feat == "Moving Averages":
                    df, ma_windows = add_moving_averages(df)
                    if ma_windows:
                        all_windows_applied.extend(ma_windows)
                elif feat == "Rolling Std Dev":
                    df, std_window = compute_rolling_stddev(df, target_col='Close')
                    all_windows_applied.append(std_window)
                elif feat == "Technical Indicators":
                    df, ti_windows = add_technical_indicators(df)
                    if ti_windows:
                        all_windows_applied.extend(ti_windows)
                elif feat == "Lag Prices":
                    df = add_lag_prices(df)
                    all_windows_applied.extend([1, 2, 3, 4, 5, 6]) # Assuming these are the lags
                elif feat == "Lagged Returns":
                    df = add_lagged_returns(df)
                    all_windows_applied.extend([1, 2, 3, 4, 5, 6]) # Assuming these are the lags
                elif feat == "Volume Features":
                    df = add_volume_features(df)
                    all_windows_applied.extend([7, 14, 21, 28, 35]) # Example window sizes for volume features
                    all_windows_applied.append(9) # Example for a specific volume feature window
                elif feat == "Time Features":
                    df = add_time_features(df)
                elif feat == "Price Structure Features":
                    df = add_price_features(df)
                else:
                    st.warning(f"Feature block '{feat}' not explicitly handled for window size tracking. Applying with default call.")
                    df = FEATURE_FUNCTIONS[feat](df)

                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Feature engineering function '{feat}' returned a non-DataFrame object (type: {type(df)}).")

            if all_windows_applied:
                st.session_state.max_calculated_feature_window = max(st.session_state.max_calculated_feature_window, max(all_windows_applied))
            else:
                st.session_state.max_calculated_feature_window = max(st.session_state.max_calculated_feature_window, 1)

            st.info(f"Calculated Max Feature Window (from applied features): {st.session_state.max_calculated_feature_window}")

        st.subheader("🧹 Handling Missing Values and Infinite Values...")

        # --- Check and Replace Infinity Values ---
        infinity_count_before_replace = np.isinf(df).sum().sum()
        st.write(f"Infinity values found (before replacement): **{infinity_count_before_replace}**")
        if infinity_count_before_replace > 0:
            st.write("Columns with infinity values:")
            st.dataframe(np.isinf(df).sum()[np.isinf(df).sum() > 0])
            st.write("Replacing infinity values with NaNs...")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            st.write(f"Total Infinity values after replacement: **{np.isinf(df).sum().sum()}**") # Should be 0

        # Calculate NaNs after feature engineering (and infinity replacement) ---
        nans_after_feat_eng = df.isna().sum().sum()
        st.write(f"1. Total NaNs after feature engineering (and infinity conversion): **{nans_after_feat_eng}**")
        st.write("   Columns with NaNs (and their counts) at this stage:")
        st.dataframe(df.isna().sum()[df.isna().sum() > 0])
        st.markdown("---")

        # Account for the largest window size by trimming rows ---
        largest_window_size = st.session_state.max_calculated_feature_window
        st.write(f"2. Accounting for largest window size ({largest_window_size}) by trimming rows...")
        if largest_window_size > 0:
            original_rows_before_trim = len(df)
            df = df.iloc[largest_window_size:].copy()
            st.success(f"Trimmed first {largest_window_size} rows. Rows remaining: {len(df)}")
        else:
            st.info("No rows trimmed as max window size is 0.")

        # --- Calculate remaining NaNs after window size trimming (Part of previous step) ---
        remaining_nans_after_trimming = df.isna().sum().sum()
        st.write(f"   Remaining NaNs after trimming rows: **{remaining_nans_after_trimming}**")
        st.write("   Columns with NaNs (and their counts) after trimming:")
        nan_cols_after_trimming = df.columns[df.isna().any()]
        if not nan_cols_after_trimming.empty:
            st.dataframe(df[nan_cols_after_trimming].isna().sum().sort_values(ascending=False))
        else:
            st.info("No columns contain NaNs after trimming rows.")
        st.markdown("---")

        # --- Step 3: Use the imputation function ---
        st.write("3. Applying custom `impute_features()` function...")
        df = impute_features(df)
        st.success("Custom imputation applied.")
        st.markdown("---")

        # --- Step 4: Calculate remaining NaNs after imputation function ---
        remaining_nans_after_imputation = df.isna().sum().sum()
        st.write(f"4. Remaining NaNs after imputation function: **{remaining_nans_after_imputation}**")
        if remaining_nans_after_imputation > 0:
            st.write("   Columns still with NaNs after imputation:")
            st.dataframe(df.isna().sum()[df.isna().sum() > 0])
        else:
            st.info("No NaNs remain after imputation.")
        st.markdown("---")

        # --- Step 5: If there are remaining NaNs, drop those rows ---
        final_df = df.copy() # Work on a copy for this final step if needed
        if remaining_nans_after_imputation > 0:
            st.write("5. NaNs still remain after imputation. Dropping rows with any remaining NaNs...")
            rows_before_final_drop = len(final_df)
            final_df.dropna(axis=0, inplace=True)
            rows_after_final_drop = len(final_df)
            st.success(f"Dropped {rows_before_final_drop - rows_after_final_drop} rows with remaining NaNs.")
        else:
            st.info("5. No remaining NaNs to drop after imputation.")

        # --- Final Data Quality Check ---
        final_nan_count = final_df.isna().sum().sum()
        final_infinity_count = np.isinf(final_df).sum().sum()

        st.subheader("✅ Final Data Quality Check:")
        st.write(f"Final NaN count: **{final_nan_count}**")
        st.write(f"Final Infinity count: **{final_infinity_count}**")

        if final_nan_count == 0 and final_infinity_count == 0:
            st.success("All NaNs and Infinity values successfully handled! Data is ready for modeling.")
        else:
            st.error("🚨 WARNING: Some NaNs or Infinity values still remain after final processing. Please review the data or imputation logic.")
            if final_nan_count > 0:
                st.dataframe(final_df.isna().sum()[final_df.isna().sum() > 0])
            if final_infinity_count > 0:
                st.dataframe(np.isinf(final_df).sum()[np.isinf(final_df).sum() > 0])


        engineered_cols = [col for col in final_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        if engineered_cols:
            st.subheader("🧬 Engineered Features (Tail)")
            st.dataframe(final_df[engineered_cols].tail())
        else:
            st.info("No new engineered features were added based on your selection.")

        st.session_state.df_loaded = final_df.copy() # Store the final, cleaned DataFrame
        st.success("Data loaded, features engineered, and missing/infinite values handled successfully!")

    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure the ticker is valid and check your internet connection.")
        st.session_state.df_loaded = None

st.sidebar.markdown("---") # Visual separator in sidebar

# --- Model Selection and Predict Button (Back in Sidebar) ---
st.sidebar.subheader("🤖 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ["Naive Forecast", "ARIMA Model", "ARIMAX Model", "XGBoost Model", "LSTM Model"], # Added LSTM Model
    index=0,
    key="sidebar_model_selector"
)

# No LSTM window size slider here as requested

if st.sidebar.button("Predict", key="sidebar_predict_button"):
    if st.session_state.df_loaded is None or st.session_state.df_loaded.empty:
        st.warning("Please load and preprocess data first before running a model.")
    else:
        st.subheader(f"🚀 Running {selected_model}...")
        try:
            if selected_model == "Naive Forecast":
                results_df, actual_values, naive_predictions = run_naive_forecast(st.session_state.df_loaded)

                st.success("Naive Forecast completed!")
                st.write("### Naive Forecast Results:")
                st.dataframe(results_df)

                if actual_values and naive_predictions:
                    st.write("### Naive Forecast: Actual vs Predicted (First Point of Each Test Split)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values (y_test[0])', color='blue', alpha=0.7)
                    ax.scatter(range(len(naive_predictions)), naive_predictions, label='Naive Predictions (y_train[-1])', color='red', alpha=0.7)
                    ax.plot(range(len(actual_values)), actual_values, color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
                    ax.plot(range(len(naive_predictions)), naive_predictions, color='red', linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

                    ax.set_title('Naive Forecast: Actual vs. Predicted per TimeSeriesSplit Fold')
                    ax.set_xlabel('Forecast Fold Index')
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("No forecast points to plot from Naive Model. Check data or splits.")

            elif selected_model == "ARIMA Model":
                st.info("Running ARIMA model. This includes hyperparameter tuning and walk-forward validation, which may take a few minutes depending on data size and CPU.")
                
                progress_text = "ARIMA model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                
                my_bar.progress(10, text="Starting ARIMA hyperparameter tuning (this can be slow)...")
                
                results_df, actual_values, predicted_values, arima_order = run_arima_forecast(st.session_state.df_loaded)
                
                my_bar.progress(100, text="ARIMA model training complete!")

                st.success("ARIMA Model completed!")
                st.write("### ARIMA Model Results:")
                st.dataframe(results_df)

                if actual_values and predicted_values:
                    st.write("### ARIMA Forecast: Actual vs Predicted (First Point of Each Test Split)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values (y_test[0])', color='blue', alpha=0.7)
                    ax.scatter(range(len(predicted_values)), predicted_values, label='ARIMA Predictions', color='green', alpha='0.7')
                    ax.plot(range(len(actual_values)), actual_values, color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
                    ax.plot(range(len(predicted_values)), predicted_values, color='green', linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

                    ax.set_title(f'ARIMA Forecast (Order: {arima_order}): Actual vs. Predicted per TimeSeriesSplit Fold')
                    ax.set_xlabel('Forecast Fold Index')
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("No forecast points to plot from ARIMA Model. Check data or splits.")

            elif selected_model == "ARIMAX Model":
                st.info("Running ARIMAX model. This includes hyperparameter tuning and walk-forward validation, which can be slower than ARIMA depending on data size and CPU.")
                
                progress_text = "ARIMAX model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                
                my_bar.progress(10, text="Starting ARIMAX hyperparameter tuning (this can be slow)...")
                
                results_df, actual_values, predicted_values, arimax_order = run_arimax_forecast(st.session_state.df_loaded)
                
                my_bar.progress(100, text="ARIMAX model training complete!")

                st.success("ARIMAX Model completed!")
                st.write("### ARIMAX Model Results:")
                st.dataframe(results_df)

                if actual_values and predicted_values:
                    st.write("### ARIMAX Forecast: Actual vs Predicted (First Point of Each Test Split)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values (y_test[0])', color='blue', alpha=0.7)
                    ax.scatter(range(len(predicted_values)), predicted_values, label='ARIMAX Predictions', color='purple', alpha=0.7)
                    ax.plot(range(len(actual_values)), actual_values, color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
                    ax.plot(range(len(predicted_values)), predicted_values, color='purple', linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

                    ax.set_title(f'ARIMAX Forecast (Order: {arimax_order}): Actual vs. Predicted per TimeSeriesSplit Fold')
                    ax.set_xlabel('Forecast Fold Index')
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("No forecast points to plot from ARIMAX Model. Check data or splits.")

            elif selected_model == "XGBoost Model":
                st.info("Running XGBoost model with GridSearchCV for hyperparameter tuning. This process can be slow depending on the data size, number of splits, and parameter grid.")
                
                progress_text = "XGBoost model training and tuning in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                
                my_bar.progress(10, text="Starting XGBoost hyperparameter tuning (this can take a while)...")
                
                results_df, actual_values, predicted_values, xgb_best_params = run_xgb_forecast(st.session_state.df_loaded)
                
                my_bar.progress(100, text="XGBoost model training and tuning complete!")

                st.success("XGBoost Model completed!")
                st.write("### XGBoost Model Results:")
                st.dataframe(results_df)
                # st.write(f"**Best Hyperparameters Found**: `{xgb_best_params}`") # Display best params

                if actual_values and predicted_values:
                    st.write("### XGBoost Forecast: Actual vs Predicted (First Point of Each Test Split)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values', color='blue', alpha=0.7)
                    ax.scatter(range(len(predicted_values)), predicted_values, label='XGBoost Predictions', color='orange', alpha=0.7)
                    ax.plot(range(len(actual_values)), actual_values, color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
                    ax.plot(range(len(predicted_values)), predicted_values, color='orange', linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

                    ax.set_title(f'XGBoost Forecast (Best Params: {xgb_best_params}): Actual vs. Predicted per TimeSeriesSplit Fold')
                    ax.set_xlabel('Forecast Fold Index')
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("No forecast points to plot from XGBoost Model. Check data or splits.")

            # LSTM Model Integration
            elif selected_model == "LSTM Model":
                # The window_size is fixed within run_lstm_forecast (defaulting to 35)
                
                st.info("Running LSTM model. This involves neural network training and walk-forward validation, which can be computationally intensive and may take significant time.")
                
                progress_text = "LSTM model training in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                
               
                my_bar.progress(10, text="Starting LSTM training...")
                
                # Call the LSTM forecast function
                results_df, actual_values, predicted_values = run_lstm_forecast(
                    st.session_state.df_loaded
                )
                
                my_bar.progress(100, text="LSTM model training complete!")

                st.success("LSTM Model completed!")
                st.write("### LSTM Model Results:")
                st.dataframe(results_df)

                if actual_values and predicted_values:
                    st.write("### LSTM Forecast: Actual vs Predicted (First Point of Each Test Split)")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values', color='blue', alpha=0.7)
                    ax.scatter(range(len(predicted_values)), predicted_values, label='LSTM Predictions', color='teal', alpha=0.7)
                    ax.plot(range(len(actual_values)), actual_values, color='blue', linestyle='--', alpha=0.5, label='Actual (Connects Points)')
                    ax.plot(range(len(predicted_values)), predicted_values, color='teal', linestyle=':', alpha=0.5, label='Predicted (Connects Points)')

                    # Get window_size from the function's default or its internal logic if you wish to display it
                    # For now, it's 35 as per the function signature.
                    lstm_window_size_display = 35 
                    ax.set_title(f'LSTM Forecast (Window Size: {lstm_window_size_display}): Actual vs. Predicted per TimeSeriesSplit Fold')
                    ax.set_xlabel('Forecast Fold Index')
                    ax.set_ylabel('Stock Price')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("No forecast points to plot from LSTM Model. Check data or splits.")

        except Exception as e:
            st.error(f"Error running {selected_model} model: {e}")
            st.exception(e) # Display full traceback for debugging

st.sidebar.markdown("---") # End of sidebar
