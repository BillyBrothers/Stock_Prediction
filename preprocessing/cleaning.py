import pandas as pd
import numpy as np
from scipy import stats

def clean_stock_data(df: pd.DataFrame, z_thresh: float = 3.0):

    """
    Clean stock data by and removing NaNs, outliers, and duplicates.
    
    Parameters:
    df (pd.Dataframe): Original stock dataframe
    z thresh (float): Z-score threshold for outlier detection

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # drop duplicate rows
    df = df[~df.index.duplicated(keep='first')]

    # drop rows with NaN Values
    df = df.dropna()

    # detect and remove outleirs using Z-score (for numberic columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < z_thresh).all(axis=1)]
    return df