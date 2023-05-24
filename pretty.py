import acquire as acq
import prepare as prep
import knear as k
import plotly.express as px
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

def decision_tree(X_train_scaled, X_validate_scaled, y_train, y_validate):
    """
    This function trains a decision tree classifier on the provided training data, and evaluates its performance on the
    validation data for different values of the 'max_depth' hyperparameter. It then generates a plot of the training and
    validation accuracy scores as a function of 'max_depth', and returns a DataFrame containing these scores.

    Parameters:
    - X_train (pandas.DataFrame): A DataFrame containing the features for the training data.
    - X_validate (pandas.DataFrame): A DataFrame containing the features for the validation data.
    - y_train (pandas.Series): A Series containing the target variable for the training data.
    - y_validate (pandas.Series): A Series containing the target variable for the validation data.

    Returns:
    - scores_df (pandas.DataFrame): A DataFrame containing the training and validation accuracy scores, as well as the
      difference between them, for different values of the 'max_depth' hyperparameter.
    """
    # get data
    scores_all = []
    for x in range(1,20):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
    
        tree.fit(X_train_scaled, y_train)
        train_acc = tree.score(X_train_scaled,y_train)
        val_acc = tree.score(X_validate_scaled, y_validate)
        score_diff = train_acc - val_acc
        scores_all.append([x, train_acc, val_acc, score_diff])
    
    scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc','val_acc','score_diff'])
    
    # Plot the results
    sns.set_style('whitegrid')
    plt.plot(scores_df['max_depth'], scores_df['train_acc'], label='Train score')
    plt.plot(scores_df['max_depth'], scores_df['val_acc'], label='Validation score')
    plt.fill_between(scores_df['max_depth'], scores_df['train_acc'], scores_df['val_acc'], alpha=0.2, color='gray')
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.show()

    return scores_df

def the_chosen_one(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Trains a K-Nearest Neighbors classifier on the provided training data with a pre-selected number of neighbors and 
    evaluates the classifier on the test data.

    Parameters:
    - X_train_scaled (pandas.DataFrame): DataFrame containing the scaled features for the training data.
    - X_test_scaled (pandas.DataFrame): DataFrame containing the scaled features for the test data.
    - y_train (pandas.Series): Series containing the target variable for the training data.
    - y_test (pandas.Series): Series containing the target variable for the test data.

    Returns:
    - test_acc (float): Accuracy score of the trained K-Nearest Neighbors classifier on the test data.
    - knn (KNeighborsClassifier): Trained K-Nearest Neighbors classifier object.
    """

    knn = KNeighborsClassifier(n_neighbors=19)
    knn.fit(X_train_scaled, y_train)
    knn.score(X_test_scaled, y_test)

    return knn.score(X_test_scaled, y_test)

def random_forest_scores(X_train_scaled, y_train, X_validate_scaled, y_validate):
    """
    Trains and evaluates a random forest classifier with different combinations of hyperparameters. The function takes in 
    training and validation datasets, and returns a dataframe summarizing the model performance on each combination of 
    hyperparameters.

    Parameters:
    -----------
    X_train : pandas DataFrame
        Features of the training dataset.
    y_train : pandas Series
        Target variable of the training dataset.
    X_validate : pandas DataFrame
        Features of the validation dataset.
    y_validate : pandas Series
        Target variable of the validation dataset.

    Returns:
    --------
    df : pandas DataFrame
        A dataframe summarizing the model performance on each combination of hyperparameters.
    """
    #define variables
    train_scores = []
    validate_scores = []
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10]
    max_depth_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    
    for min_samples_leaf, max_depth in zip(min_samples_leaf_values, max_depth_values):
        rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth,random_state=123)
        rf.fit(X_train_scaled, y_train)
        train_score = rf.score(X_train_scaled, y_train)
        validate_score = rf.score(X_validate_scaled, y_validate)
        train_scores.append(train_score)
        validate_scores.append(validate_score)
       
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - validate_score for train_score, validate_score in zip(train_scores, validate_scores)]
    
    #Put results into a dataframe
    df = pd.DataFrame({
        'min_samples_leaf': min_samples_leaf_values,
        'max_depth': max_depth_values,
        'train_score': train_scores,
        'validate_score': validate_scores,
        'diff_score': diff_scores})
     
    # Set plot style
    sns.set_style('whitegrid')
 
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(max_depth_values, train_scores, label='train', marker='o', color='blue')
    plt.plot(max_depth_values, validate_scores, label='validation', marker='o', color='orange')
    plt.fill_between(max_depth_values, train_scores, validate_scores, alpha=0.2, color='gray')
    plt.xticks([2,4,6,8,10],['Leaf 9 and Depth 2','Leaf 7 and Depth 4','Leaf 5 and Depth 6','Leaf 3 and Depth 8','Leaf 1and Depth 10'], rotation = 45)
    plt.xlabel('min_samples_leaf and max_depth', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Random Forest Classifier Performance', fontsize=18)
    plt.legend(fontsize=12)
    plt.show()
    
    return df

def plot_logistic_regression(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''
    Trains multiple logistic regression models with different regularization strengths (C) on the given training
    data, and plots the resulting train and validation scores against C values. The optimal value of C is marked
    by a vertical red dashed line, and the associated difference between the train and validation scores is shown
    in the plot legend.

    Parameters:
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    X_validate : array-like of shape (n_samples, n_features)
        The validation input samples.
    y_train : array-like of shape (n_samples,)
        The target values for training.
    y_validate : array-like of shape (n_samples,)
        The target values for validation.

    Returns:
    df1 : pandas DataFrame
        A table containing the C, train_score, validate_score, and diff_score values for each model.
    '''
    train_scores = []
    val_scores = []
    c_values = [.01, .1, 1, 10 , 100, 1000]
    for c in c_values:
        logit = LogisticRegression(C=c, random_state=123)
        logit.fit(X_train_scaled, y_train)
        train_score = logit.score(X_train_scaled, y_train)
        val_score = logit.score(X_validate_scaled, y_validate)
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    # Calculate the difference between the train and validation scores
    diff_scores = [train_score - val_score for train_score, val_score in zip(train_scores, val_scores)]
     
    # Put results into a list of tuples
    results = list(zip(c_values, train_scores, val_scores, diff_scores))
    # Convert the list of tuples to a Pandas DataFrame
    df1 = pd.DataFrame(results, columns=['C', 'train_score', 'validate_score', 'diff_score'])
    

    # Plot the results
    plt.plot(c_values, train_scores, label='Train score')
    plt.plot(c_values, val_scores, label='Validation score')
    min_diff_idx = np.abs(diff_scores).argmin()
    min_diff_c = results[min_diff_idx][0]
    min_diff_score = results[min_diff_idx][3]
    plt.axvline(min_diff_c, linestyle='--', linewidth=2, color='red', label=f'min diff at C={min_diff_c} (diff={min_diff_score:.3f})')
    plt.fill_between(c_values, train_scores, val_scores, alpha=0.2, color='gray')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Accuracy vs C')
    plt.legend()
    plt.show()

    return df1

def plot_target(train):
    """
    Plots a count bar plot of the 'quality' column in the provided dataset.

    Parameters:
    train (pandas.DataFrame): The training dataset containing the 'quality' column.

    Returns:
    None
    """
    sns.countplot(x=train.quality)
    plt.title('Target = Quality')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.show()

def variables_for_clustering(X_train_scaled, X_validate_scaled, X_test_scaled):
    """
    Selects specific independent variables for clustering using the K-means algorithm and returns them for further use in validation and testing.

    Parameters:
    X_train_scaled (pandas.DataFrame): Scaled training data containing all relevant features.
    X_validate_scaled (pandas.DataFrame): Scaled validation data containing all relevant features.
    X_test_scaled (pandas.DataFrame): Scaled test data containing all relevant features.

    Returns:
    tuple: A tuple containing three pandas.DataFrame objects: X, x1, and x2.
        - X: Independent variables selected from the scaled training data (X_train_scaled).
        - x1: Independent variables selected from the scaled validation data (X_validate_scaled).
        - x2: Independent variables selected from the scaled test data (X_test_scaled).

    This function takes the scaled versions of the training, validation, and test data and selects specific independent variables
    for clustering using the K-means algorithm. The chosen variables are 'alcohol', 'residual_sugar', and 'density'. These variables
    are selected from the respective input datasets, and the resulting subsets of data are returned as a tuple.

    """
    # define independent variables for k-means, carry those on to validate and test
    X = X_train_scaled[['alcohol','residual_sugar','density']]
    x1 = X_validate_scaled[['alcohol','residual_sugar','density']]
    x2 = X_test_scaled[['alcohol','residual_sugar','density']]
    
    return X, x1, x2

def elbow_graph_for_k(X):
    """
    Visualizes the elbow method for selecting the optimal number of clusters (k) using the K-means algorithm.

    Parameters:
    X (array-like): Input data for clustering.

    Returns:
    None

    This function plots a graph showing the change in inertia (sum of squared distances from each point to its assigned centroid)
    as the number of clusters (k) increases. The inertia values are calculated using the K-means algorithm with values of k ranging
    from 2 to 11 (inclusive). The graph helps in identifying the optimal value of k by identifying the 'elbow' point where the
    rate of decrease in inertia significantly slows down.

    The graph is created using the 'seaborn-whitegrid' style and has a figure size of 9x6. Each value of k is fitted to the
    K-means algorithm, and the inertia value is calculated for each cluster configuration. The inertia values are then plotted
    on the y-axis, with the corresponding k values marked on the x-axis. The x-axis ticks are set to range from 2 to 11. The
    x-axis is labeled as 'k', and the y-axis is labeled as 'inertia'. The title of the graph is 'Change in inertia as k increases'.

    Example usage:
    >>> data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> elbow_graph_for_k(data)
    """
# view elbow method selection for k
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

def cluster_creation1(X, x1, x2, X_train_scaled, X_validate_scaled, X_test_scaled):
    """
    Create clusters based on the given data and assign cluster labels to the datasets.

    Args:
        X (pandas.DataFrame): The data used for clustering.
        x1 (pandas.DataFrame): Validation data used for prediction.
        x2 (pandas.DataFrame): Test data used for prediction.
        X_train_scaled (pandas.DataFrame): Scaled training data.
        X_validate_scaled (pandas.DataFrame): Scaled validation data.
        X_test_scaled (pandas.DataFrame): Scaled test data.

    Returns:
        X_train_scaled (pandas.DataFrame): Training data with cluster labels.
        X_validate_scaled (pandas.DataFrame): Validation data with cluster labels.
        X_test_scaled (pandas.DataFrame): Test data with cluster labels.

    """

    # MAKE the thing
    kmeans = KMeans(n_clusters=3)

    # FIT the thing
    kmeans.fit(X)

    # USE (predict using) the thing 
    kmeans.predict(X)
    kmeans.predict(x1) #validate
    kmeans.predict(x2) #test

    # make new column names in X_train_scaled, X_validate_scaled, X_test_scale and X dataframe
    X_train_scaled['sugar_alcohol_density'] = kmeans.predict(X)

    X_validate_scaled['sugar_alcohol_density'] = kmeans.predict(x1)

    X_test_scaled['sugar_alcohol_density'] = kmeans.predict(x2)

    X['sugar_alcohol_density'] = kmeans.predict(X)

    # doing the things
    X_train_scaled['sugar_alcohol_density'] = X_train_scaled.sugar_alcohol_density

    #rename using map
    X_train_scaled['sugar_alcohol_density'] = X_train_scaled.sugar_alcohol_density.map({
    0: 'low_sugar, low_alcohol , med_density',
    1: 'high_alcohol, low_sugar, low_density',
    2: 'low_alcohol, high_sugar, high_density'
    })

    # Scatter plot of unscaled data with hue for cluster
    plt.figure(figsize=(13, 8))
    sns.scatterplot(x='alcohol', y='residual_sugar', data=X_train_scaled, hue='density', palette='viridis')

    # Plot cluster centers for 'alcohol', 'residual sugar', and 'density'
    cluster_centers = kmeans.cluster_centers_ 
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='d', s=200, linewidths=2, label='Cluster Centers')

    # Scatter plot of 'sugar_alcohol_density' clusters
    sns.scatterplot(x='alcohol', y='residual_sugar', data=X_train_scaled, hue='sugar_alcohol_density', palette='Set1', alpha=0.5)

    plt.title('Clustering of Residual Sugar, Alcohol, and Density')
    plt.xlabel('Alcohol')
    plt.ylabel('Residual Sugar')
    plt.legend()
    plt.show()

    return X_train_scaled, X_validate_scaled, X_test_scaled

def cluster_vs_target(X_train_scaled, y_train):
    """
    Visualize the relationship between clusters and the target variable (quality).

    Args:
        X_train_scaled (pandas.DataFrame): The scaled training data.
        y_train (pandas.Series): The target variable for the training data.

    Returns:
        None

    """
    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Customize the barplot
    sns.barplot(data=X_train_scaled, x='sugar_alcohol_density', y=y_train, palette='coolwarm')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set labels and title with appropriate font size
    plt.xlabel('Sugar, Alcohol, Density', fontsize=12)
    plt.ylabel('Quality', fontsize=12)
    plt.title('Clusters vs. Quality', fontsize=14)

    # Remove spines (top and right)
    sns.despine()

    # Adjust spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

def rename_for_modeling(X_train_scaled):
   """
    Rename the 'sugar_alcohol_density' column values in the training data for modeling.

    Args:
        X_train_scaled (pandas.DataFrame): The scaled training data.

    Returns:
        X_train_scaled (pandas.DataFrame): The training data with renamed 'sugar_alcohol_density' column values.

    """
    # rename using map
   X_train_scaled['sugar_alcohol_density'] = X_train_scaled.sugar_alcohol_density.map({
     'low_sugar, low_alcohol , med_density': 0,
     'high_alcohol, low_sugar, low_density': 1,
     'low_alcohol, high_sugar, high_density': 2})
   return X_train_scaled

def select_features_for_modeling(X_train_scaled, X_validate_scaled, X_test_scaled):
    """
    Select specific features for modeling from the scaled datasets.

    Args:
        X_train_scaled (pandas.DataFrame): The scaled training data.
        X_validate_scaled (pandas.DataFrame): The scaled validation data.
        X_test_scaled (pandas.DataFrame): The scaled test data.

    Returns:
        X_train_scaled (pandas.DataFrame): The training data with selected features.
        X_validate_scaled (pandas.DataFrame): The validation data with selected features.
        X_test_scaled (pandas.DataFrame): The test data with selected features.

    """
    X_train_scaled = X_train_scaled[['fixed_acidity', 'volatile_acidity','citric_acid','chlorides'
                     ,'free_sulfur_dioxide','ph','sulphates'
                     ,'bound_sulfur_dioxide','White','sugar_alcohol_density']]
    X_validate_scaled = X_validate_scaled[['fixed_acidity', 'volatile_acidity','citric_acid','chlorides'
                     ,'free_sulfur_dioxide','ph','sulphates'
                     ,'bound_sulfur_dioxide','White','sugar_alcohol_density']]
    X_test_scaled = X_test_scaled[['fixed_acidity', 'volatile_acidity','citric_acid','chlorides'
                     ,'free_sulfur_dioxide','ph','sulphates'
                     ,'bound_sulfur_dioxide','White','sugar_alcohol_density']]
    return X_train_scaled, X_validate_scaled, X_test_scaled

def individual_cluster_plot(X_train_scaled):
    """
    Generate an individual cluster plot based on the scaled training data.

    Args:
        X_train_scaled (pandas.DataFrame): The scaled training data.

    Returns:
        None

    """
    sns.set_style("whitegrid")

    g = sns.relplot(x='alcohol', y='residual_sugar', data=X_train_scaled, hue='density', col='sugar_alcohol_density')
    g.set_axis_labels('Alcohol', 'Residual Sugar')
    plt.show()
    return