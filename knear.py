import itertools
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def k_nearest2(X_train_scaled, y_train, X_validate_scaled, y_validate):
    """
    Trains and evaluates KNN models for different values of k and plots the results.

    Parameters:
    -----------
    X_train: array-like, shape (n_samples, n_features)
        Training input samples.
    y_train: array-like, shape (n_samples,)
        Target values for the training input samples.
    X_validate: array-like, shape (n_samples, n_features)
        Validation input samples.
    y_validate: array-like, shape (n_samples,)
        Target values for the validation input samples.

    Returns:
    --------
    results: pandas DataFrame
        Contains the train and validation accuracy for each value of k.
    """
    metrics = []
    train_score = []
    validate_score = []
    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_score.append(knn.score(X_train_scaled, y_train))
        validate_score.append(knn.score(X_validate_scaled, y_validate))
        diff_score = train_score[-1] - validate_score[-1]
        metrics.append({'k': k, 'train_score': train_score[-1], 'validate_score': validate_score[-1], 'diff_score': diff_score})

    baseline_accuracy = (y_train == 6).mean()

    results = pd.DataFrame.from_records(metrics)

    # modify the last few lines of the function
    # drop the diff_score column before plotting
    results_for_plotting = results.drop(columns=['diff_score'])
    with sns.axes_style('whitegrid'):
        ax = results_for_plotting.set_index('k').plot(figsize=(16,9))
    plt.ylabel('Accuracy')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xticks(np.arange(0,21,1))   
    min_diff_idx = np.abs(results['diff_score']).argmin()
    min_diff_k = results.loc[min_diff_idx, 'k']
    min_diff_score = results.loc[min_diff_idx, 'diff_score']
    ax.axvline(min_diff_k, linestyle='--', linewidth=2, color='red', label=f'min diff at k={min_diff_k} (diff={min_diff_score:.3f})')
    plt.fill_between(results['k'], train_score, validate_score, alpha=0.2, color='gray', where=(results['k'] > 0))    
    plt.title('K Nearest Neighbor', fontsize=18)
    plt.legend()
    plt.show()
    
    return results



    
    
