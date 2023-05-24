import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

import env
import os
import wrangle as w

def remove_outliers(df, exclude_columns=None):
    """
    Remove outliers from a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        exclude_columns (list): Columns to exclude from outlier detection and removal. Default is None.

    Returns:
        df_clean (pandas.DataFrame): The cleaned DataFrame with outliers removed.
        summary (pandas.DataFrame): Summary of removed outliers for each column.

    """
    # Copy the input dataframe to preserve the original data
    df_clean = df.copy()

    # Dictionary to store outlier information for each column
    outlier_info = {}

    # Columns to exclude from outlier detection and removal
    if exclude_columns is None:
        exclude_columns = []

    # Iterate over each column in the dataframe
    for col in df.columns:
        # Skip columns with non-numeric data and excluded columns
        if not np.issubdtype(df[col].dtype, np.number) or col in exclude_columns:
            continue

        # Compute quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        # Compute interquartile range
        IQR = Q3 - Q1

        # Define upper and lower bounds
        upper_bound = Q3 + (1.5 * IQR)
        lower_bound = Q1 - (1.5 * IQR)

        # Find outliers
        outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]

        # Remove outliers from the clean dataframe
        df_clean = df_clean[~df_clean.index.isin(outliers.index)]

        # Store outlier information for the column
        outlier_info[col] = {
            'Upper Bound': upper_bound,
            'Lower Bound': lower_bound,
            'Outliers Removed': len(outliers)
        }

    # Generate plots for before and after removing outliers
    for col in df.columns:
        # Skip non-numeric and excluded columns
        if not np.issubdtype(df[col].dtype, np.number) or col in exclude_columns:
            continue

        plt.figure(figsize=(12, 6))

        # Boxplot before removing outliers
        plt.subplot(1, 2, 1)
        plt.title(f'{col} - Before Outlier Removal')
        sns.boxplot(data=df, y=col)

        # Boxplot after removing outliers
        plt.subplot(1, 2, 2)
        plt.title(f'{col} - After Outlier Removal')
        sns.boxplot(data=df_clean, y=col)

        plt.tight_layout()
        plt.show()

    # Summary of removed outliers
    summary = pd.DataFrame.from_dict(outlier_info, orient='index')
    summary['Outliers Removed'] = summary['Outliers Removed'].astype(int)
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Column'}, inplace=True)

    return df_clean, summary