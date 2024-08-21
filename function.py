import re

# Import the string dictionary that we'll use to remove punctuation
import string 
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def clean_text_syntax(text):
    """
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    """

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.
    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text_syntax(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text

def find_best_hyperparametered_model(model_name, model, X_train_vect, y_train):


    if model_name == "SVM":
        print("SVM Parameter tuning Started ...")

        # Define the parameter distribution for SVM
        param_dist = {
            'C': np.logspace(-3, 3, 10),  # Regularization parameter
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
            'gamma': ['scale', 'auto']  # Kernel coefficient
        }

        # Instantiate the RandomizedSearchCV object
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

        # Fit on the training data
        random_search.fit(X_train_vect, y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)
        print("SVM Parameter tuning Finished.")

        return grid_search
    
    elif model_name == "Naive_Bayes":
        print("Naive Bayes Parameter tuning Started ...")

        # Define the parameter grid for Naive Bayes
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0]  # Smoothing parameter
        }

        # Instantiate the GridSearchCV object
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

        # Fit on the training data
        grid_search.fit(X_train_vect, y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        print("Naive Bayes Parameter tuning Finished.")

        return grid_search

    elif model_name == "Logistic":
        print("Logistic Parameter tuning Started ...")

        # Define the parameter distribution for Logistic Regression
        param_dist = {
            'C': np.logspace(-3, 3, 10),  # Regularization strength
            'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization type
            'solver': ['lbfgs', 'liblinear', 'saga']  # Solvers for optimization
        }

        # Instantiate the RandomizedSearchCV object
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

        # Fit on the training data
        random_search.fit(X_train_vect, y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)
        print("Logistic Regression Parameter tuning Finished.")

        return grid_search

    elif model_name == "RandomForest":
        print("Random Forest tuning Started ...")

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],  # Number of trees
            'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of each tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        }

        # Instantiate the GridSearchCV object
        grid_search = GridSearchCV(model , param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(X_train_vect, y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        print("Random Forest Parameter tuning Finished.")

        return grid_search
    
    elif model_name == "XGBoost":
        print("XGBoost tuning Started ...")

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }

        # Instantiate the GridSearchCV object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

        # Fit the grid search to the data
        grid_search.fit(X_train_vect, y_train)

        # Print the best parameters and the best score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        print("XGBoost Parameter tuning Finished.")

        return grid_search

