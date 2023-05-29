# wine_project

# Goal:
- Predict the quality of wine while incorporating unsupervised/supervised learning techniques.

# Description:
- Using unsupervised (exploration with clustering) / supervised machine learning (classification modeling) we want to try predicting the quality of wine. We will be using features from the wine dataset, and clustering models to build clusters within our notebook. Using stats tests and visualizations, including clustering we will attempt to single out the drivers of quality. We will then attempt to create a MLM that predicts the quality of wine.
Finally, we will present the findings to our peers in a storytelling format using a live presentation.


# Data Dictionary:

| Feature                | Data Type | Description                                       |
|------------------------|-----------|---------------------------------------------------|
| fixed_acidity          | float     | Fixed acidity level of the wine $(g/dm^3)$        |
| volatile_acidity       | float     | Volatile acidity level of the wine $(g/dm^3)$     |
| citric_acid            | float     | Citric acid content of the wine $(g/dm^3)$        |
| residual_sugar         | float     | Residual sugar content of the wine $(g/dm^3)$     |
| chlorides              | float     | Chloride content of the wine $(g/dm^3)$           |
| free_sulfur_dioxide    | float     | Free sulfur dioxide content of the wine $(mg/L)$  |
| total_sulfur_dioxide   | float     | Total sulfur dioxide content of the wine $(mg/L)$ |
| density                | float     | Density of the wine $(g/cm^3)$                    |
| pH                     | float     | pH value of the wine                              |
| sulphates              | float     | Sulphate content of the wine $(g/dm^3)$           |
| alcohol                | float     | Alcohol content of the wine (% vol)               |
| bound_sulfur_dioxide   | float     | Bound sulfur dioxide content of the wine $(mg/L)$ |
| White                  | int       | Wine type: 1 for white wine, 0 for red wine       |
| alcohol_sugar_density  | int       | Cluster of alcohol_sugar_density                  |
| quality                | int       | Subjective human taste test rating                |


# Project Planning:

- Aquire data from (https://data.world/food/wine-quality) and make it a local .csv file so we can both collectivly work from the same data
- Prep the data for exploration: splitting/testing/modeling
- Explore the data for significant features and answer questions we have about the data set
- Perform clustering techniques for further exploration of features/feature engineering
- Model on train,validate data sets; Test datasets to see how our features can help predict the quality of wine
- Create a presentation


# Initial Questions:
- Is there a relationship between Quality (Categorical) vs. Residual sugar (Continuous)?
- Is there a relationship between Quality (Categorical) vs. Free Sulfur Dioxide (Continuous)?
- Is there a relationship between Quality (Categorical) vs. Total Sulfur Dioxide (Continuous)?
- Is there a relationship between Quality (Categorical) vs. Bound Sulfur Dioxide (Continuous)?
- Is there a relationship between Quality (Categorical) vs. Density (Continuous)?


# Steps to reproduce:
- Download data from: https://data.world/food/wine-quality
- Use function in acquire.py: 'get_wine_data()'
    - ***Change*** the following two lines of code to indicate where your files are located. 
    - white_file_path = '/Users/***director where files are located***/codeup-data-science/wine_project/winequality_white.csv'
    - red_file_path = '/Users/***director where files are located***/codeup-data-science/wine_project/winequality_red.csv'
- Prep the data 
    - Use function in wrangle.py to remove outliers
- Use functions in prepare.py to:
    - Split data into train validate and test datasets using the 60,20,20 method.
- Explore the data using visualizations and stats tests.
- Continue exploring the data further using the k means clustering method.
- Using the best features run classification models on the train and validate to find the best model.
- We used KNN on our test dataset.
- Read over the outputs and form a conclusion and summary with recommendations.

# Summary/Conclusion:
- The following data analysis pipeline categories were accomplished (Acquire, Prepare, Explore(with clustering), Model) we found that alcohol, sugar, and density were high drivers of predicting quality.

- We selected the KNN (K nearest neighbors). The decision was made based on highest accuracy score with least amount of difference between train and validate data sets.
- The KNN performed slightly better than baseline, so we may need to look at other factors that are not available in the dataset like aroma, legs, and taste. 

# Recommendations:
- All of the selected features contributed to the predicting the quality of wine.  
    - Collect more data on red wines. The data set was 1 sided with 75% of the data being white wines. 
        - Diversity in the data would create a better data environment for prediction accuracy. 
        - Higher alcohol and lower sugar seems to drive quality, not by much but a noticeable difference. 

# Next steps:
- With more time, further combination of clusters may create features that create a model with more predictive power.
    - Analyzing the relationships between multiple features in relation to quality could improve the model. 
    - Even though the selected features affect quality, their relationships to other factors may direct where we want to focus efforts in predicting quality.






```python

```
