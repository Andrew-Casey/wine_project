import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

import env
import os
import wrangle as w

def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 

def get_zillow():
    """
    Retrieves extended Zillow data from a MySQL database and returns it as a Pandas DataFrame.

    The function connects to a MySQL database using the provided database URL
    and executes a query to fetch extended Zillow data for single-family residential properties
    from the year 2017. The data includes the tax value, square footage, county code, number of bedrooms,
    number of bathrooms, lot size, year built, and pool count. The additional information includes
    story type and construction type, if available.

    Returns:
        pandas.DataFrame: Extended Zillow data for single-family residential properties in 2017.

    Example:
        >>> df = get_zillow2()
        >>> df.head()
           Tax_Value    Sqft  County  Bedrooms  Bathrooms  Lot_Size  Year_Built  Pool
        0   360170.0  1316.0  6037.0       3.0        2.0    12647.0      1923.0   1.0
        1   585529.0  1458.0  6059.0       3.0        2.0     9035.0      1970.0   0.0
        2   119906.0  1421.0  6037.0       2.0        1.0     7500.0      1911.0   0.0
        3   244880.0  2541.0  6037.0       4.0        3.0     8777.0      2003.0   0.0
        4   434551.0  1491.0  6059.0       3.0        2.0     6388.0      1955.0   0.0
    """
    url = env.get_db_url('zillow')

    query = """
    select *,
	CAST(latitude / 1e6 AS DECIMAL(10, 6)) AS latitude_dd,
    CAST(longitude / 1e6 AS DECIMAL(10, 6)) AS longitude_dd
    from properties_2017
    join predictions_2017 using (parcelid)
    join propertylandusetype using (propertylandusetypeid)
    left join architecturalstyletype using (architecturalstyletypeid)
    left join airconditioningtype using (airconditioningtypeid)
    left join buildingclasstype using (buildingclasstypeid)
    left join heatingorsystemtype using (heatingorsystemtypeid)
    left join storytype using (storytypeid)
    left join typeconstructiontype using (typeconstructiontypeid)
    left join unique_properties using (parcelid)
    where transactiondate Like '2017%%'
    ;
    """
    filename = 'zillow3.csv'
    df = check_file_exists(filename, query, url)

    return df

def wrangle_zillow():
    """
    Performs data wrangling on extended Zillow dataset.

    This function retrieves the extended Zillow dataset using the `get_zillow2()` function and performs
    several data wrangling steps. It changes FIPS codes to county names, creates dummy variables for the
    county column, fills null values in the 'Pool' column with 0, drops remaining null values, and handles
    outliers for the tax value, square footage, number of bedrooms, number of bathrooms, lot size, and
    year built.

    Returns:
        pandas.DataFrame: Wrangled extended Zillow dataset.

    Example:
        >>> df = wrangle_zillow2()
        >>> df.head()
           Tax_Value    Sqft County  Bedrooms  Bathrooms  Lot_Size  Year_Built  Pool  LA  Orange  Ventura
        0   360170.0  1316.0     LA       3.0        2.0   12647.0      1923.0   1.0   1       0        0
        1   585529.0  1458.0  Orange      3.0        2.0    9035.0      1970.0   0.0   0       1        0
        2   119906.0  1421.0     LA       2.0        1.0    7500.0      1911.0   0.0   1       0        0
        3   244880.0  2541.0     LA       4.0        3.0    8777.0      2003.0   0.0   1       0        0
        4   434551.0  1491.0  Orange      3.0        2.0    6388.0      1955.0   0.0   0       1        0
    """
    # Load extended Zillow database
    df = w.get_zillow()

    # Change FIPS codes to county name
    df['fips'] = df['fips'].replace([6037.0, 6059.0, 6111.0], ['LA', 'Orange', 'Ventura']).astype(str)

    # Create dummy variables for the county column
    dummy_df = pd.get_dummies(df['fips'], drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)

    # Fill pool and fireplace null values with 0
    df.poolcnt = df.poolcnt.fillna(0)
    df.fireplacecnt = df.fireplacecnt.fillna(0)

    # set df to only single family residential homes
    df = df[df.propertylandusedesc == 'Single Family Residential']

    # use function to filter data by dropping columns missing 50% of the data and dropping rows that dont retain 75% of the data
    df = w.handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75)      

    # use function to remove large columns of missing values.
    df = w.data_prep(df,
    cols_to_remove=['heatingorsystemtypeid', 'buildingqualitytypeid','propertyzoningdesc',
                   'unitcnt','heatingorsystemdesc'], 
    prop_required_columns= .50,
    prop_required_rows=0.75)

    # Drop all remaining nulls
    df = df.dropna()

    # Handle duplicate values in parcel id
    df = df.sort_values(by=['transactiondate'], ascending=False)
    #make dataframe of duplicates
    duplicates = df[df.duplicated('parcelid', keep=False)].sort_values('parcelid')

    #make a mask to remove
    mask = duplicates.groupby('parcelid')['transactiondate'].transform(max) != duplicates['transactiondate']

    #apply mask and save as filtered_df
    filtered_df = df[~df.index.isin(duplicates[mask].index)]

    #reset name to df
    df = filtered_df

    df = df.drop(columns=['latitude','longitude','regionidcounty','id','id.1'
                     ,'finishedsquarefeet12','calculatedbathnbr','finishedsquarefeet12'
                     ,'rawcensustractandblock','roomcnt','propertylandusetypeid','assessmentyear'
                     ,'fullbathcnt'])

    return df

# set nulls_by_row function
def nulls_by_row(df, index_id = 'parcelid'):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({
                    'num_cols_missing': num_missing,
                    'percent_cols_missing': pct_miss
                    })
    rows_missing = df.merge(rows_missing, 
                        left_index=True,
                        right_index=True).reset_index()[['parcelid', 'num_cols_missing','percent_cols_missing']]
    
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

# set nulls_by_col function
def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

# set get object data type columns
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols


#set get numeric data type columns
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

# set handle missing values function
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df

# set remove_columns function
def remove_columns(df, cols_to_remove):
    """
    This function will:
    - take in a df and list of columns
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=cols_to_remove)
    return df

# set data prep functions
def data_prep(df, cols_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):

    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = w.remove_columns(df, cols_to_remove)
    df = w.handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

def remove_outliers(df, exclude_columns=None):
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

# Compute the mean(logerror) by zipcode
#zipcode_means = df.groupby('regionidzip')['logerror'].mean()

# Compute the overall mean(logerror)
#overall_mean = df['logerror'].mean()

# Perform t-tests for each zipcode against the overall mean
#significant_zipcodes = []
#alpha = 0.05  # significance level

#for zipcode, mean in zipcode_means.items():
   # t_stat, p_value = stats.ttest_1samp(df[df['regionidzip'] == zipcode]['logerror'], overall_mean)
   # if p_value < alpha:
   #     significant_zipcodes.append(zipcode)

# Output the results
#print("Mean(logerror) by regionidzip:")
#print(zipcode_means)
#print("\nOverall mean(logerror):", overall_mean)
#print("\nSignificant zip codes with errors different from expected:")
#print(significant_zipcodes)