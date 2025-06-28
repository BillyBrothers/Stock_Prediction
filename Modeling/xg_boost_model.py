import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV # NEW: Import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def run_xgb_forecast(msft_df, target_col='Close', n_splits=5):
    """
    Runs an XGBoost time series forecast with hyperparameter tuning using GridSearchCV
    and walk-forward validation.
    Features and target are scaled, then inverse-scaled for final evaluation.

    Parameters:
        msft_df (pd.DataFrame): DataFrame containing the target_col and features.
        target_col (str): The name of the target column (e.g., 'Close').
        n_splits (int): Number of splits for TimeSeriesSplit cross-validation.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with the XGBoost Mean Squared Error and Best Params.
            - list: List of actual values (original scale) for the first step of each test split.
            - list: List of predicted values (original scale) for the first step of each test split.
            - dict: Dictionary of the best parameters found by GridSearchCV.
    """
    if len(msft_df) < n_splits + 1:
        raise ValueError(f"Not enough data points ({len(msft_df)}) for {n_splits} splits. Consider reducing n_splits or providing more data.")

    X = msft_df.drop(columns=[target_col])
    y = msft_df[target_col]

    # Ensure X only contains numeric columns before scaling
    X_numeric = X.select_dtypes(include=np.number)
    if X_numeric.empty:
        raise ValueError("No numeric features found for XGBoost. Please ensure your DataFrame contains numeric features besides the target column.")

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Fit and transform X and y
    X_scaled = X_scaler.fit_transform(X_numeric)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # --- NEW: Hyperparameter Tuning with GridSearchCV ---
    xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42) # Base model for GridSearchCV

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    # For GridSearchCV's internal cross-validation, using a TimeSeriesSplit.
    # It's often good to use a smaller number of splits here than for the main walk-forward
    # to speed up the tuning process, e.g., min(n_splits, 3).
    # However, the user explicitly asked for n_splits=5 for CV in GridSearchCV, so we use that.
    grid_search_cv = TimeSeriesSplit(n_splits=100) 

    # verbose=1 will print progress to the console (not directly in Streamlit app output)
    # n_jobs=-1 will use all available CPU cores
    grid = GridSearchCV(xgb_base, param_grid, cv=grid_search_cv, scoring='neg_mean_squared_error', verbose=1)
    
    # Fit GridSearchCV on the full scaled data to find the best parameters
    grid.fit(X_scaled, y_scaled.ravel()) # .ravel() converts y to a 1D array as expected by XGBoost

    best_xgb_params = grid.best_params_
    print("Best XGBoost params:", best_xgb_params) # This will print to the Streamlit console/terminal

    # --- Walk-forward forecasting using the best estimator found ---
    tscv = TimeSeriesSplit(n_splits=n_splits)
    actual_values_orig_scale, predicted_values_orig_scale = [], []

    for train_idx, test_idx in tscv.split(msft_df): # Use msft_df for splitting to get correct indices
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

        # Initialize a new XGBoost model for each fold with the best parameters found by GridSearchCV
        # This ensures proper walk-forward validation where each fold's model is trained from scratch
        # with the globally best hyperparameters.
        xgb_fold_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_xgb_params)
        
        # Fit the model on the training data of the current fold
        xgb_fold_model.fit(X_train, y_train.ravel())

        # Predict only the first step of the test set (as per walk-forward validation strategy)
        y_pred_scaled = xgb_fold_model.predict(X_test[[0]]) # X_test[[0]] keeps it 2D for single sample

        # Inverse transform predictions and actual values back to original scale
        # y_scaler.inverse_transform expects 2D array, and returns 2D array.
        # We then take [0][0] to get the single scalar value.
        y_pred_orig_scale = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
        y_test_orig_scale = y_scaler.inverse_transform(y_test[[0]])[0][0]

        actual_values_orig_scale.append(y_test_orig_scale)
        predicted_values_orig_scale.append(y_pred_orig_scale)

    mse = mean_squared_error(actual_values_orig_scale, predicted_values_orig_scale)
    
    # Include best parameters in the results DataFrame
    results_df = pd.DataFrame({
        "XGBoost_MSE": [mse],
        "XGBoost_Best_Params": [str(best_xgb_params)] # Convert dict to string for display
    })

    # NEW: Return best_xgb_params along with other results
    return results_df, actual_values_orig_scale, predicted_values_orig_scale, best_xgb_params
