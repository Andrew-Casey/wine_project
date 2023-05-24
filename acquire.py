#imports
import numpy as np
import pandas as pd
from pydataset import data
import os
import env as e

import os
import pandas as pd

def check_file_exists(file_path):
    """
    Check if a file exists in the local directory.

    Parameters:
    -----------
    file_path : str
        File path to check for existence.

    Returns:
    --------
    bool:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def get_wine_data():
    """
    Read white and red wine data from CSV files in the local directory,
    combine them into a single dataframe, add a column to identify whether
    each row corresponds to red or white wine, and write the combined
    dataframe to a CSV file.

    Returns:
    --------
    pandas.DataFrame:
        Combined dataframe containing white and red wine data with an additional 'Type' column.
    """
    # Define the file path for the combined CSV
    combined_csv_path = '/Users/andrewcasey/codeup-data-science/wine_project/combined_wine_data.csv'

    # Check if the combined CSV file exists
    if check_file_exists(combined_csv_path):
        print(f"Combined wine data CSV file found at {combined_csv_path}. Loading...")
        # Read the combined CSV file into a dataframe
        df = pd.read_csv(combined_csv_path)
    else:
        print("Combined wine data CSV file not found. Creating new combined dataframe...")
        # Get the file paths for white and red wine CSVs
        white_file_path = '/Users/andrewcasey/codeup-data-science/wine_project/winequality_white.csv'
        red_file_path = '/Users/andrewcasey/codeup-data-science/wine_project/winequality_red.csv'

        # Read the white and red wine CSVs into dataframes
        white_df = pd.read_csv(white_file_path)
        red_df = pd.read_csv(red_file_path)

        # Create new columns to identify whether the wines are red or white
        white_df['Type'] = 'White'
        red_df['Type'] = 'Red'

        # Combine the two dataframes into one
        combined_df = pd.concat([white_df, red_df], ignore_index=True)

        # Create dummy variables for the Type column
        dummy_df = pd.get_dummies(combined_df['Type'], drop_first=True)
        df = pd.concat([combined_df, dummy_df], axis=1)

        # Write the combined dataframe to a CSV file
        df.to_csv(combined_csv_path, index=False)
        print(f"Combined wine data CSV file created at {combined_csv_path}.")

    return df

def get_wine():
    """
    Opens csv from file.

    Parameters:
    -----------
    file_path : file should be in same folder as your current working directory

    Returns:
    --------
    combined data frame of red and white wine
    """
    df = pd.read_csv('combined_wine_data.csv')

    df['bound_sulfur_dioxide'] = (df.total_sulfur_dioxide - df.free_sulfur_dioxide)
    
    return df

    