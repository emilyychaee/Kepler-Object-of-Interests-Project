# Kepler-Object-of-Interests-Project

""" 
  Use various machine learning models to accurately predict the disposition of Kepler Objects of Interest based   on characteristics. 

"""

import pandas as pd
import numpy as np
import sklearn.metrics as skm
import fairlearn.metrics as fm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def main():
    # import the data and read the file
    # pd.set_option('display.width', None)
    file_path = "cumulative_2023.11.07_13.44.30.csv"
    df_koi = pd.read_csv(file_path, skiprows=41)

    # # wrangle data and drop duplicates and attributes
    df_koi = df_koi.drop_duplicates()
    print(df_koi)

    df_koi = df_koi[
        ['koi_disposition', 'koi_pdisposition', 'koi_period', 'koi_eccen', 'koi_duration', 'koi_prad', 'koi_sma',
         'koi_incl', 'koi_teq', 'koi_dor', 'koi_steff', 'koi_srad', 'koi_smass']]

    df_koi = df_koi.dropna()

    print(df_koi)
    X = df_koi.drop(columns='koi_pdisposition', axis=1)
    y = df_koi['koi_pdisposition']

    # categorical data in X, encode any categorical data to numeric data
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023, stratify=y)

    param_grids = {
        'Logistic Regression': {},
        'K Nearest Neighbors': {'n_neighbors': np.arange(1, 1.5)},
        'Decision Tree': {'criterion': ['entropy', 'gini'],
                          'max_depth': np.arange(3, 16),
                          'min_samples_leaf': np.arange(1, 11)},
        'SVM': {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}
    }

    # Logistic Regression
    print("Running cross-validation for Logistic Regression without PCA...")
    logistic_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('Logistic Regression', LogisticRegression(max_iter=1000))
    ])

    logistic_grid_search = GridSearchCV(logistic_pipeline, param_grids['Logistic Regression'], cv=5, scoring='accuracy',
                                        n_jobs=-1)
    logistic_grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {logistic_grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {logistic_grid_search.best_score_}\n")

    # K Nearest Neighbors
    print("Running cross-validation for K Nearest Neighbors without PCA")
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('K Nearest Neighbors', KNeighborsClassifier())
    ])

    knn_grid_search = GridSearchCV(knn_pipeline, param_grids['K Nearest Neighbors'], cv=5, scoring='accuracy',
                                   n_jobs=-1)
    knn_grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {knn_grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {knn_grid_search.best_score_}\n")

    # Decision Tree
    print("Running cross-validation for Decision Tree without PCA")
    tree_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('Decision Tree', DecisionTreeClassifier())
    ])

    tree_grid_search = GridSearchCV(tree_pipeline, param_grids['Decision Tree'], cv=5, scoring='accuracy', n_jobs=-1)
    tree_grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {tree_grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {tree_grid_search.best_score_}\n")

    # SVM
    print("Running cross-validation for SVM without PCA")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('SVM', SVC())
    ])

    svm_grid_search = GridSearchCV(svm_pipeline, param_grids['SVM'], cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid_search.fit(X_train, y_train)

    print(f"Best hyperparameters: {svm_grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {svm_grid_search.best_score_}\n")

    # Evaluate the final optimized SVM model on the test set
    final_svm_model = svm_grid_search.best_estimator_
    y_svm_pred = final_svm_model.predict(X_test)

    # Visualize confusion matrix for SVM
    conf_matrix_svm = confusion_matrix(y_test, y_svm_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_svm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (SVM)')
    plt.colorbar()

    classes_svm = np.unique(y)
    tick_marks_svm = np.arange(len(classes_svm))
    plt.xticks(tick_marks_svm, classes_svm, rotation=45)
    plt.yticks(tick_marks_svm, classes_svm)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.show()

    # Display classification report for SVM
    class_report_svm = classification_report(y_test, y_svm_pred)
    print("Classification Report (SVM):\n", class_report_svm)


if __name__ == '__main__':
    main()
