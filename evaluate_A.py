import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def plot_residuals(y, yhat):
    """
    Plots the residuals between the actual values (y) and the predicted values (yhat).

    Parameters:
        y (pandas Series): The actual values.
        yhat (pandas Series): The predicted values.

    Returns:
        None

    The function calculates the residuals by subtracting the predicted values (yhat) from the actual values (y).
    It also calculates the baseline by taking the mean of the actual values.
    The function then plots the residuals against the actual values using a scatter plot.
    The baseline is displayed as a horizontal line.
    The x-axis represents the actual values, and the y-axis represents the residuals.
    The title of the plot is set as 'Residual Plot'.
    """
    residuals = y - yhat
    baseline = y.mean()
    # baseline
    plt.axhline(baseline, ls=':', color='black')
    sns.scatterplot(y = residuals, x = y, hue = y)
    #plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def regression_errors(y, yhat):
    """
    Calculates regression evaluation metrics based on the predicted values (yhat) and the actual values (y).

    Parameters:
        y (pandas Series): The actual values.
        yhat (pandas Series): The predicted values.

    Returns:
        Tuple of:
            SSE (float): The sum of squared errors.
            ESS (float): The explained sum of squares.
            TSS (float): The total sum of squares.
            MSE (float): The mean squared error.
            RMSE (float): The root mean squared error.

    The function calculates the errors by subtracting the predicted values (yhat) from the actual values (y).
    It then calculates the squared errors.
    The function proceeds to calculate the sum of squared errors (SSE), the explained sum of squares (ESS),
    the total sum of squares (TSS), the mean squared error (MSE), and the root mean squared error (RMSE).
    The calculated metrics are returned as a tuple of floats.
    """
    errors = y - yhat
    squared_errors = errors ** 2
    
    SSE = np.sum(squared_errors)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    MSE = np.mean(squared_errors)
    RMSE = np.sqrt(MSE)
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """
    Calculates regression evaluation metrics based on the baseline mean prediction and the actual values (y).

    Parameters:
        y (pandas Series): The actual values.

    Returns:
        Tuple of:
            SSE (float): The sum of squared errors.
            MSE (float): The mean squared error.
            RMSE (float): The root mean squared error.

    The function calculates the baseline mean prediction by taking the mean of the actual values (y).
    It then calculates the errors by subtracting the baseline mean prediction from the actual values (y).
    The function proceeds to calculate the squared errors.
    It calculates the sum of squared errors (SSE), the mean squared error (MSE), and the root mean squared
    error (RMSE) based on the squared errors.
    The calculated metrics are returned as a tuple of floats.
    """
    baseline_prediction = np.mean(y)
    errors = y - baseline_prediction
    squared_errors = errors ** 2
    
    SSE = np.sum(squared_errors)
    MSE = np.mean(squared_errors)
    RMSE = np.sqrt(MSE)
    
    return SSE, MSE, RMSE

def better_than_baseline(y, yhat, baseline):
    """
    Determines if the model's sum of squared errors (SSE) is lower than the baseline's SSE.

    Parameters:
        y (pandas Series): The actual values.
        yhat (pandas Series): The predicted values from the model.
        baseline (float or int): The baseline value to compare against.

    Returns:
        bool: True if the model's SSE is lower than the baseline's SSE, False otherwise.

    The function calculates the sum of squared errors (SSE) for both the model's predictions (yhat) and the baseline value.
    It compares the SSE values and returns True if the model's SSE is lower than the baseline's SSE,
    and False otherwise.
    """
    sse_model = np.sum((y - yhat) ** 2)
    sse_baseline = np.sum((y - baseline) ** 2)

    return sse_model < sse_baseline

#recursive feature elimination
def rfe(X, y, k):
    """
    Performs Recursive Feature Elimination (RFE) to select the top 'k' features based on the given data.

    Parameters:
        X (pandas DataFrame): The feature matrix.
        y (pandas Series): The target variable.
        k (int): The number of features to select.

    Returns:
        List of str: The names of the selected features.

    The function uses a LinearRegression estimator and Recursive Feature Elimination (RFE) to select the top 'k' features
    from the feature matrix (X) based on their importance in predicting the target variable (y).
    It fits the RFE selector on the data and retrieves the mask of selected features.
    The names of the selected features are extracted using the mask and returned as a list of strings.
    """
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    mask = selector.support_
    selected_features = X.columns[mask]
    return selected_features

#K best feature selection
def select_kbest(X, y, k):
    """
    Performs SelectKBest feature selection to select the top 'k' features based on the given data.

    Parameters:
        X (pandas DataFrame): The feature matrix.
        y (pandas Series): The target variable.
        k (int): The number of features to select.

    Returns:
        List of str: The names of the selected features.

    The function uses the SelectKBest feature selection method with the f_regression scoring function to select
    the top 'k' features from the feature matrix (X) based on their importance in predicting the target variable (y).
    It fits the selector on the data and retrieves the mask of selected features.
    The names of the selected features are extracted using the mask and returned as a list of strings.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected_features = X.columns[mask]
    return selected_features

def run_regression1(df, target_var):
    """
    Runs linear regression on different combinations of features in the given DataFrame and target variable.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the feature variables.
        target_var (pandas Series): The target variable to predict.

    Returns:
        Tuple: A tuple containing the best combination of features with the highest R^2 score.

    The function runs linear regression on different combinations of features from the DataFrame (df) to predict
    the target variable (target_var). It scales the feature variables using StandardScaler and fits a LinearRegression model.
    For each combination of features, it calculates the root mean squared error (RMSE) and R^2 score.
    The function keeps track of the best combination with the lowest RMSE and the best combination with the highest R^2 score.
    It also plots a regression plot of the predicted values versus the actual values.

    After iterating through all combinations, the function prints the best RMSE and R^2 score with their respective feature combinations.
    Finally, it returns the best combination of features with the highest R^2 score as a tuple.
    """
    columns = df.columns.tolist()
    combinations = [combo for r in range(1, len(columns) + 1) for combo in itertools.combinations(columns, r)]
    best_rmse = float('inf')
    best_rmse_combo = None
    best_r2 = float('-inf')
    best_r2_combo = None

    for combo in combinations:
        X = df[list(combo)]
        y = target_var

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = LinearRegression().fit(X_scaled, y)
        y_pred = reg.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_combo = combo

        if r2 > best_r2:
            best_r2 = r2
            best_r2_combo = combo

        print(f'RMSE: {rmse:.2f}, R^2: {r2:.2f} for {combo}')

    sns.regplot(x=y_pred, y=y, line_kws={'color':'red'}, scatter_kws={'alpha':0.06})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(f'Best RMSE: {best_rmse:.2f} for {best_rmse_combo}')
    print(f'Best R^2: {best_r2:.2f} for {best_r2_combo}')
    return best_r2_combo

def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def baseline_model(train, y_train):
    """
    Creates a baseline model using the mean of the target variable and evaluates its performance.

    Parameters:
        train (pandas DataFrame): The training data containing the feature variables.
        y_train (pandas Series): The target variable for the training data.

    Returns:
        pandas DataFrame: A DataFrame containing the evaluation metrics of the baseline model.

    The function creates a baseline model by setting the predicted value as the mean of the target variable (y_train).
    It calculates the root mean squared error (RMSE) and R^2 score of the baseline model using the y_train values
    and an array filled with the mean value. The RMSE and R^2 score are added to a DataFrame for comparison.

    Additionally, the function prints the baseline value and returns the DataFrame with the evaluation metrics.
    """
    #set baseline
    baseline = round(y_train.mean(),2)

    #make an array to send into my mean_square_error function
    baseline_array = np.repeat(baseline, len(train))

    # Evaluate the baseline rmse and r2
    rmse, r2 = metrics_reg(y_train, baseline_array)

    # add results to a dataframe for comparison
    metrics_df = pd.DataFrame(data=[
    {
        'model':'Baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    
    # print baseline
    baseline = round(y_train.mean(),2)
    print(f' Baseline mean is : {baseline}')
    return metrics_df

def multiple_regression(X_train_scaled, X_validate_scaled,y_validate,y_train, metrics_df):
    """
    Performs multiple regression using Recursive Feature Elimination (RFE) and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_validate_scaled (pandas DataFrame): The scaled feature variables of the validation data.
        y_validate (pandas Series): The target variable for the validation data.
        y_train (pandas Series): The target variable for the training data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the multiple regression model.

    The function performs multiple regression using the Recursive Feature Elimination (RFE) technique. It fits
    a Linear Regression model on the RFE-transformed training data (X_train_rfe) and makes predictions on the
    RFE-transformed validation data (X_val_rfe).

    The function evaluates the model's performance by calculating the root mean squared error (RMSE) and R^2 score
    using the `metrics_reg` function with the predicted values (pred_val_OLS) and the y_validate values.

    The RMSE and R^2 score are added to the provided metrics DataFrame (metrics_df) for comparison. The updated
    metrics DataFrame is returned.
    """
    #### make it
    OLS = LinearRegression()
    #use Recursive Feature Eliminations
    rfe = RFE(OLS, n_features_to_select=5)
    #fit it
    rfe.fit(X_train_scaled, y_train)
    #use it on train
    X_train_rfe = rfe.transform(X_train_scaled)
    #use it on validate
    X_val_rfe = rfe.transform(X_validate_scaled)

    # build model for the top features
    #fit the thing
    OLS.fit(X_train_rfe, y_train)
    #use the thing (make predictions)
    
    pred_val_OLS = OLS.predict(X_val_rfe)

    # Evaluate Validate
    rmse, r2 = metrics_reg(y_validate, pred_val_OLS)    

    #add to my metrics df
    metrics_df.loc[1] = ['Multiple Regression', rmse, r2]

    return metrics_df
    
def LassoLars_model(X_train_scaled, X_validate_scaled, y_train, y_validate, metrics_df):
    """
    Performs LassoLars regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_validate_scaled (pandas DataFrame): The scaled feature variables of the validation data.
        y_train (pandas Series): The target variable for the training data.
        y_validate (pandas Series): The target variable for the validation data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the LassoLars model.

    The function performs LassoLars regression by fitting a LassoLars model on the scaled training data
    (X_train_scaled) and making predictions on the scaled validation data (X_validate_scaled).

    The function evaluates the model's performance by calculating the root mean squared error (RMSE) and R^2 score
    using the `metrics_reg` function with the predicted values (pred_val_lars) and the y_validate values.

    The RMSE and R^2 score are added to the provided metrics DataFrame (metrics_df) for comparison. The updated
    metrics DataFrame is returned.
    """
    #make it
    lars = LassoLars(normalize=False, alpha=1)
    #fit it
    lars.fit(X_train_scaled, y_train)
    #use it
    pred_val_lars = lars.predict(X_validate_scaled)
    #validate
    rmse, r2 = metrics_reg(y_validate, pred_val_lars)
    #add to my metrics df
    metrics_df.loc[2] = ['LassoLars', rmse, r2]
    
    return metrics_df

def polynomial_regression(X_train_scaled, X_validate_scaled, X_test_scaled, y_validate, y_train, metrics_df ):
    """
    Performs polynomial regression and evaluates the model's performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_validate_scaled (pandas DataFrame): The scaled feature variables of the validation data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_validate (pandas Series): The target variable for the validation data.
        y_train (pandas Series): The target variable for the training data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the polynomial regression model.

    The function performs polynomial regression by creating polynomial features of degree 3 using the `PolynomialFeatures`
    transformer. It fits and transforms the scaled training data (X_train_scaled) to obtain the new set of features
    (X_train_degree3). It then transforms the scaled validation data (X_validate_scaled) and test data (X_test_scaled)
    using the same transformer.

    The function fits a linear regression model (pr) on the transformed training data (X_train_degree3) and makes
    predictions on both the training data (pred_pr) and the validation data (pred_val_pr). Additionally, it makes
    predictions on the test data (pred_val_test).

    The function evaluates the model's performance by calculating the root mean squared error (RMSE) and R^2 score using
    the `metrics_reg` function with the predicted values (pred_val_pr) and the y_validate values.

    The calculated RMSE and R^2 score for the validation data are added to the provided metrics DataFrame (metrics_df)
    for comparison. Finally, the updated metrics DataFrame is returned.
    """
    # make the polynomial features to get a new set of features
  
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_val_degree3 = pf.transform(X_validate_scaled)

    X_test_degree3 = pf.transform(X_test_scaled)

    #make it
    pr = LinearRegression()
    #fit it
    pr.fit(X_train_degree3, y_train)
    #use it
    pred_pr = pr.predict(X_train_degree3)
    pred_val_pr = pr.predict(X_val_degree3)
    pred_val_test=pr.predict(X_test_degree3)   

    #validate
    rmse, r2 = metrics_reg(y_validate, pred_val_pr) 

    #add to my metrics df
    metrics_df.loc[3] = ['Polynomial Regression(PR)', rmse, r2]
    return metrics_df

def Generalized_Linear_Model(X_train_scaled, X_validate_scaled, y_train, y_validate, metrics_df):
    """
    Fits a Generalized Linear Model (GLM) and evaluates its performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_validate_scaled (pandas DataFrame): The scaled feature variables of the validation data.
        y_train (pandas Series): The target variable for the training data.
        y_validate (pandas Series): The target variable for the validation data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        pandas DataFrame: The updated metrics DataFrame with the evaluation metrics of the GLM.

    The function fits a Generalized Linear Model (GLM) using the TweedieRegressor with power=1 and alpha=0. It fits the
    GLM on the scaled training data (X_train_scaled) and the corresponding target variable (y_train).

    The function uses the fitted GLM to make predictions on both the training data (pred_glm) and the validation data
    (pred_val_glm).

    The function evaluates the GLM's performance by calculating the root mean squared error (RMSE) and R^2 score using
    the `metrics_reg` function with the predicted values (pred_val_glm) and the y_validate values.

    The calculated RMSE and R^2 score for the validation data are added to the provided metrics DataFrame (metrics_df)
    for comparison. Finally, the updated metrics DataFrame is returned.
    """
    #make it
    glm = TweedieRegressor(power=1, alpha=0)
    #fit it
    glm.fit(X_train_scaled, y_train)
    #use it
    pred_glm = glm.predict(X_train_scaled)
    pred_val_glm = glm.predict(X_validate_scaled)

    # validate
    rmse, r2 = metrics_reg(y_validate, pred_val_glm)

    #add to my metrics df
    metrics_df.loc[4] = ['Generalized Linear Model', rmse, r2]
    
    return metrics_df

def polynomial_regression_test(X_train_scaled, X_test_scaled, y_test, y_train, metrics_df ):
    """
    Performs polynomial regression on the test data and evaluates its performance.

    Parameters:
        X_train_scaled (pandas DataFrame): The scaled feature variables of the training data.
        X_test_scaled (pandas DataFrame): The scaled feature variables of the test data.
        y_test (pandas Series): The target variable for the test data.
        y_train (pandas Series): The target variable for the training data.
        metrics_df (pandas DataFrame): A DataFrame to store the evaluation metrics.

    Returns:
        Tuple[pandas DataFrame, numpy ndarray]: A tuple containing the updated metrics DataFrame with the evaluation
        metrics of the polynomial regression on test data, and the array of predicted target values (pred_pr).

    The function performs polynomial regression on the test data using the features from the training data. It uses the
    PolynomialFeatures transformer with degree=3 to create a new set of polynomial features.

    The function fits and transforms the scaled training data (X_train_scaled) using the PolynomialFeatures transformer
    to obtain X_train_degree3, which represents the training data with polynomial features.

    The function transforms the scaled test data (X_test_scaled) using the already fitted PolynomialFeatures transformer
    to obtain X_test_degree3, which represents the test data with polynomial features.

    The function initializes a LinearRegression model (pr), fits it on the training data with polynomial features
    (X_train_degree3) and the corresponding target variable (y_train), and uses it to make predictions on the test data
    (pred_pr).

    The function evaluates the performance of the polynomial regression on the test data by calculating the root mean
    squared error (RMSE) and R^2 score using the `metrics_reg` function (presumed to be defined elsewhere) with the
    predicted values (pred_pr) and the y_test values.

    The calculated RMSE and R^2 score for the test data are added to the provided metrics DataFrame (metrics_df) for
    comparison. Finally, the updated metrics DataFrame and the array of predicted target values (pred_pr) are returned
    as a tuple.
    """
    # make the polynomial features to get a new set of features
  
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train_scaled)

    X_test_degree3 = pf.transform(X_test_scaled)

    #make it
    pr = LinearRegression()
    #fit it
    pr.fit(X_train_degree3, y_train)
    #use it
  
    pred_pr=pr.predict(X_test_degree3)   

    #validate
    rmse, r2 = metrics_reg(y_test, pred_pr) 

    #add to my metrics df
    metrics_df.loc[5] = ['PR on test data', rmse, r2]
    return metrics_df, pred_pr

def plot_residuals(y_train, y_test, pred_pr):
    """
    Plot the comparison of predicted and actual values for polynomial regression.

    Parameters:
        y_train (pandas Series): The target variable for the training data.
        y_test (pandas Series): The target variable for the test data.
        pred_pr (numpy ndarray): The array of predicted target values.

    The function plots the actual values (y_test) against the predicted values (pred_pr) obtained from polynomial
    regression. It also includes a scatter plot of the predicted values and adds a diagonal line representing the perfect
    prediction.

    The baseline value, calculated as the mean of the training target variable (y_train), is represented by a horizontal
    dashed line. The label "Baseline" is annotated at the coordinates (65, 81) on the plot.

    The plot is titled "Comparison of Predicted and Actual Values" with the y-axis labeled as "Actual Value" and the
    x-axis labeled as "Predicted Final Value". A legend is included to identify the polynomial regression line and the
    perfect prediction line. The plot also includes a grid for better visualization.

    The function displays the plot and does not return any value.
    """

    baseline = round(y_train.mean(),2)
    #Plotting actuals vs. predicted values
    plt.scatter(pred_pr, y_test, label='Polynomial 2nd degree', alpha=0.5, color='orange', s=10)


    plt.plot(y_test, y_test, label='_nolegend_', color='purple', linestyle='--')

    plt.axhline(baseline, ls=':', color='grey')
    plt.annotate("Baseline", (65, 81), color='grey')

    plt.title("Comparison of Predicted and Actual Values")
    plt.ylabel("Actual Value in millions (USD)")
    plt.xlabel("Predicted Value in millions (USD)")


    plt.grid(True, linestyle='--', alpha=0.5) 

    plt.show()

    return

def examine_target(train):
    """
    Visualize the distribution of the target variable in the training data.

    Parameters:
        train (pandas DataFrame): The training data containing the target variable.

    The function plots a histogram of the target variable, which is represented by the 'Tax_Value' column in the training
    data. The histogram provides insights into the distribution of the target variable values.

    The plot is titled 'Target = Tax Value', with the x-axis labeled as 'Tax Value' and the y-axis labeled as 'Count'.
    The function displays the plot and does not return any value.
    """
    # View Target
    sns.histplot(train.Tax_Value)
    plt.title('Target = Tax Value')
    plt.xlabel('Tax Value in millions (USD)')
    plt.ylabel('Count')
    plt.show()

def examine_Sqft_and_TxValue(train):
    """
    Visualize the relationship between the 'Sqft' feature and the 'Tax_Value' target variable.

    Parameters:
        train (pandas DataFrame): The training data containing the 'Sqft' and 'Tax_Value' columns.

    The function creates a scatter plot with a regression line to visualize the relationship between the 'Sqft' feature and
    the 'Tax_Value' target variable in the training data.

    The scatter plot shows the data points where the x-axis represents 'Sqft' and the y-axis represents 'Tax_Value'. It also
    includes a regression line that represents the trend in the data.

    The plot is titled 'Square Feet vs. Target', with the x-axis labeled as 'Square Feet' and the y-axis labeled as
    'Tax Value'.

    The function displays the plot using `plt.show()` and does not return any value.
    """
    #Visualize Sqft and Tax_Value
    sns.set(style="darkgrid")

    # Create a scatter plot with regression line
    sns.relplot(x="Sqft", y="Tax_Value", data=train, kind="scatter")
    sns.regplot(x="Sqft", y="Tax_Value", data=train, scatter=False, color='red')

    # Show the plot
    plt.xlabel('Square Feet')
    plt.ylabel('Tax Value in millions (USD)')
    plt.title('Square Feet vs. Target')
    plt.show()

def examine_Year_Built_and_TxValue(train):
    """
    Visualize the relationship between the 'Year_Built' feature and the 'Tax_Value' target variable.

    Parameters:
        train (pandas DataFrame): The training data containing the 'Year_Built' and 'Tax_Value' columns.

    The function creates a scatter plot with a regression line to visualize the relationship between the 'Year_Built' feature
    and the 'Tax_Value' target variable in the training data.

    The scatter plot shows the data points where the x-axis represents 'Year_Built' and the y-axis represents 'Tax_Value'.
    It also includes a regression line that represents the trend in the data.

    The plot is titled 'Year Built vs. Tax Value', with the x-axis labeled as 'Year Built' and the y-axis labeled as
    'Tax Value'.

    The function displays the plot using `plt.show()` and does not return any value.
    """
    #Visualize Year_Built and Tax_Value
    sns.set(style="darkgrid")

    # Create a scatter plot with regression line
    sns.relplot(x="Year_Built", y="Tax_Value", data=train, kind="scatter")
    sns.regplot(x="Year_Built", y="Tax_Value", data=train, scatter=False, color='red')

    # Show the plot
    plt.xlabel('Year Built')
    plt.ylabel('Tax Value in millions (USD)')
    plt.title('Year Built vs. Tax Value')
    plt.show()

def examine_heat_map(train):
    """
    Generate a correlation heatmap to visualize the pairwise correlations between features in the training data.

    Parameters:
        train (pandas DataFrame): The training data containing the features.

    The function generates a correlation heatmap using the 'pearson' correlation coefficient to measure the pairwise
    correlations between features in the training data.

    The heatmap is displayed using a color map ('PRGn') where the intensity of the color represents the strength of the
    correlation. Positive correlations are displayed in green, negative correlations in purple, and no correlation in white.

    The heatmap includes numerical annotations in each cell representing the correlation coefficient. The annotations are
    formatted with two decimal places using the 'fmt=".2f"' parameter.

    The upper triangle of the heatmap is masked using 'np.triu()' to hide duplicate correlation values and focus on the
    lower triangle.

    The font size of the y-axis and x-axis tick labels is adjusted to 8 using 'plt.yticks(fontsize=8)' and 'plt.xticks(fontsize=8)'.

    The plot is titled 'Correlation heat map'.

    The function displays the heatmap using 'plt.show()' and does not return any value.
    """
    # Increase the figure size to accommodate the heatmap
    plt.figure(figsize=(10, 8))
    # Correlation heat map
    sns.heatmap(train.corr(method='pearson'), cmap='PRGn', annot=True, fmt=".2f",
            mask=np.triu(train.corr(method='pearson')))
    # Adjust the font size of the annotations
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.title('Correlation heat map')
    # Show the plot
    plt.show()

def explore_target(y_train):
    """
    Visualize the distribution of the target variable in the training data.

    Parameters:
        y_train (pandas Series): The target variable.

    The function generates a histogram to visualize the distribution of the target variable in the training data.

    The histogram displays the frequency (number of houses) on the y-axis and the value of the target variable on the x-axis.

    The x-axis label is set as "Value" using `plt.xlabel("Value")`, and the y-axis label is set as "Number of houses" using
    `plt.ylabel("Number of houses")`.

    The plot is titled 'Distribution of home values' using `plt.title('Distribution of home values')`.

    The function displays the histogram using `plt.show()` and does not return any value.
    """
    #visualize baseline
    plt.hist(y_train)
    plt.xlabel("Tax Value in millions (USD)")
    plt.ylabel("Number of houses")
    plt.title('Distribution of home values')
    plt.show()