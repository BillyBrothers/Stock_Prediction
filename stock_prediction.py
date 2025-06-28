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
    trim_nans_by_window, # Still useful to have, though we'll use iloc directly for clarity
    impute_features      # Your custom imputation function
)

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

# Sidebar layout
st.sidebar.title("📈 Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=3)
interval = st.sidebar.selectbox("Select Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo", "3mo"], index=8)

# Optional feature enhancements
apply_features = st.sidebar.checkbox("Add Engineered Features")

if apply_features:
    selected_blocks = st.sidebar.multiselect(
        "Select Feature Blocks",
        options=list(FEATURE_FUNCTIONS.keys()),
        default=["Lag Prices", "Lagged Returns", "Moving Averages", "Rolling Std Dev", "Technical Indicators", "Time Features", "Volume Features", "Price Structure Features"]
    )
else:
    selected_blocks = []

# Load button
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
# Global variable to hold the maximum window size from engineered features
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

        # --- Step 1: Calculate NaNs after feature engineering (and infinity replacement) ---
        nans_after_feat_eng = df.isna().sum().sum()
        st.write(f"1. Total NaNs after feature engineering (and infinity conversion): **{nans_after_feat_eng}**")
        st.write("   Columns with NaNs (and their counts) at this stage:")
        st.dataframe(df.isna().sum()[df.isna().sum() > 0])
        st.markdown("---")

        # --- Step 2: Account for the largest window size by trimming rows ---
        largest_window_size = st.session_state.max_calculated_feature_window
        st.write(f"2. Accounting for largest window size ({largest_window_size}) by trimming rows...")
        if largest_window_size > 0:
            original_rows_before_trim = len(df)
            df = df.iloc[largest_window_size:].copy()
            st.success(f"Trimmed first {largest_window_size} rows. Rows remaining: {len(df)}")
        else:
            st.info("No rows trimmed as max window size is 0.")

        # --- Calculate remaining NaNs after window size trimming (Part of Step 2) ---
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

st.sidebar.markdown("---")