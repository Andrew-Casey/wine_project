import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

import env
import os
import wrangle as w

def plot_variable_pairs(train):
    """
    Generate pairwise scatter plots for the variables in the given DataFrame.

    Parameters:
        train (pandas DataFrame): The DataFrame containing the variables.

    The function uses seaborn's pairplot to create pairwise scatter plots for all combinations of variables in the DataFrame.

    The argument `kind="reg"` specifies that regression lines should be plotted on the scatter plots.

    The argument `corner=True` sets the corner plot to display only the lower triangle of the pairwise scatter plots,
    resulting in a more compact visualization.

    The argument `plot_kws={'line_kws': {'color': 'red'}}` is used to set the color of the regression lines to red.

    The function displays the plot using `plt.show()` and does not return any value.
    """
    sns.set(style="ticks")
    sns.pairplot(train, kind="reg", corner = True, plot_kws={'line_kws': {'color': 'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(dataframe, categorical_var, continuous_var):
    """
    Generate multiple plots to visualize the relationship between a categorical variable and a continuous variable.

    Parameters:
        dataframe (pandas DataFrame): The DataFrame containing the data.
        categorical_var (str): The name of the categorical variable.
        continuous_var (str): The name of the continuous variable.

    The function generates three plots: a box plot, a strip plot, and a bar plot.

    The box plot displays the distribution of the continuous variable across different categories of the categorical variable.
    The x-axis represents the categorical variable, and the y-axis represents the continuous variable.

    The strip plot shows individual data points as scattered points, providing an overview of the distribution of the continuous variable for each category of the categorical variable.

    The bar plot displays the average value of the continuous variable for each category of the categorical variable.

    Each plot is displayed using `plt.show()`.

    The function does not return any value.
    """
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Box Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Strip Plot of {continuous_var} vs {categorical_var}")
    plt.show()

    # Swarm plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dataframe[categorical_var], y=dataframe[continuous_var])
    plt.xlabel(categorical_var)
    plt.ylabel(continuous_var)
    plt.title(f"Bar Plot of {continuous_var} vs {categorical_var}")
    plt.show()

  