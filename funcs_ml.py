import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (SGDRegressor, LogisticRegression, LinearRegression)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor)
# https://stackoverflow.com/questions/25336176/does-scikit-learn-include-a-naive-bayes-classifier-with-continuous-inputs
from sklearn.naive_bayes import GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score, make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split    

import ast

# https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods
from lightgbm import LGBMRegressor
import xgboost
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from operator import itemgetter

import warnings
warnings.filterwarnings('ignore')

def feature_selection_by_extra_tree_regressor(X, y):
    extra_tree_forest = ExtraTreesRegressor(random_state=42) # Initializing ExtraTreeClassRegressor
    extra_tree_forest.fit(X, y)
    ranked_feature = pd.Series(extra_tree_forest.feature_importances_, index=X.columns) # features by importance
    features = ranked_feature.sort_values(ascending=False).index
    return list(features)

def feature_selection_by_xgb(X, y):
    xgb = XGBRegressor(random_state=42) # Initializing XGBRegressor
    xgb.fit(X, y)
    ranked_feature = pd.Series(xgb.feature_importances_, index=X.columns)  # features by importance
    features = ranked_feature.sort_values(ascending=False).index
    return list(features)

feature_selection_methods= {
        'feature_selection_by_extra_tree_regressor' : feature_selection_by_extra_tree_regressor,
        'feature_selection_by_xgb' : feature_selection_by_xgb
    }


def xgb_gridsearch_cv( X, y, cv=5, scoring='neg_mean_absolute_percentage_error'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    model = XGBRegressor(random_state=42)
    param_grid = {
        'max_depth': [2,4,6,8,10],
        'learning_rate': [0.01,0.1,0.3,0.5,0.9,1],
        'n_estimators': [20, 30, 40, 50, 70, 100]
    }
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,verbose=1)
    grid_search.fit(X, y)
    model = XGBRegressor(**grid_search.best_params_, random_state=42)
    return grid_search

def lightgbm_gridsearch_cv( X, y, cv=5, scoring='neg_mean_absolute_percentage_error'):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    model = LGBMRegressor(random_state=42)
    param_grid = {
        'num_leaves': [2,4,6,8],
        'max_depth': [ 20, 30, 40, 50, 70, 100],
        'learning_rate': [0.05, 0.1, 0.2,0.5,1],
        'reg_alpha': [0, 0.01, 0.03],
    }
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,verbose=1)
    grid_search.fit(X, y)
    model = LGBMRegressor(**grid_search.best_params_, random_state=42)
    return grid_search

def test_model_iteratively_with_most_important_features(X, y, regressor_gridsearch_func):
    results = pd.DataFrame()

    for feature_selection in feature_selection_methods.keys():
        fs_method = feature_selection_methods[feature_selection]
        selected_features = fs_method(X, y)
        for no_of_features in range(1,len(selected_features)+1,25):
            print(f'Train model with {feature_selection} for {no_of_features} number of features.')
            grid_search = regressor_gridsearch_func( X[selected_features[:no_of_features]], y)
            results = pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
            results['feature_selection'] = feature_selection
            results['no_of_features'] = no_of_features
            results['selected_features'] = str(selected_features[:no_of_features])
            results = results.append(results)
    return results

def train_best_model(X, y, results, regressor_model):
    model = regressor_model(**results['params'].values[0], random_state=42)
    #feature_selection = feature_selection_methods[results['feature_selection'].values[0]]
    #no_of_features = results['no_of_features'].values[0]
    selected_features = ast.literal_eval(results['selected_features'].values[0])

    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print('mape', mean_absolute_percentage_error(y_test, y_pred))
    print('mae', mean_absolute_error(y_test, y_pred))
    print('r2', r2_score(y_test, y_pred))
    
    return y_pred

def model_using_train_test_split(X, y):
    split = int(len(X) * 0.7)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]


    results = []
    for model in [
        DummyRegressor,
        LinearRegression,
        DecisionTreeRegressor,
        KNeighborsRegressor,
        BayesianRidge,
        SVR,
        RandomForestRegressor,
        xgboost.XGBRegressor,
        LGBMRegressor,
        #CatBoostRegressor,
        SGDRegressor,
        KernelRidge,
        #ElasticNet,
        GradientBoostingRegressor
        ]:
        
        cls = model()

        print("Training:", model.__name__)
        
        cls.fit(X_train, y_train)
        
        y_true = y_test.values
        y_pred = cls.predict(X_test)
        
        r2 = metrics.r2_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
        mse=metrics.mean_squared_error(y_true, y_pred) 
            
        results.append([
                model.__name__,
                f"{r2:.3f}",
                f"{mean_absolute_error:.3f}",
                f"{mse:.3f}",
        ])
    results = sorted(results, key=itemgetter(1), reverse=True) # sort table by test_r2     
    return results


def model_using_kfold(X, y):
    for model in [
        DummyRegressor,
        LinearRegression,
        DecisionTreeRegressor,
        KNeighborsRegressor,
        BayesianRidge,
        SVR,
        RandomForestRegressor,
        xgboost.XGBRegressor,
        ]:
        
        cls = model()
        kfold = model_selection.KFold(n_splits=10) # , random_state=42
        s = model_selection.cross_val_score(cls, X, y, scoring="r2", cv=kfold)
        
        print(
        f"{model.__name__:22} r2: "
        f"{s.mean():.3f} STD: {s.std():.2f}"
        )

#max_year = int(X.index.max()[:4]) 
#min_year = int(X.index.min()[:4])
#print("max", max_year)
#print("min", min_year)
#cv_splits= max_year - min_year - 1
#print("cv_splits", cv_splits)

# reference: https://www.geeksforgeeks.org/find-average-list-python/
def mean(lst):
    return sum(lst) / len(lst)

# reference: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
def model_using_time_series_split(X, y):
    # https://goldinlocks.github.io/Time-Series-Cross-Validation/ 
    results = []
    for model in [
        DummyRegressor,
        LinearRegression,
        DecisionTreeRegressor,
        KNeighborsRegressor,
        BayesianRidge,
        SVR,
        RandomForestRegressor,
        xgboost.XGBRegressor,
        LGBMRegressor,
        #CatBoostRegressor,
        SGDRegressor,
        KernelRidge,
        #ElasticNet,
        GradientBoostingRegressor
        ]:
        
        cls = model()
        
        cv_splits = 20 # we have over 20 years of data
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # This was the older way I was doing it but it wasn't that effective..
        #scoring = ( 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error')
        #print("Train model:", model.__name__)
        #scores = cross_validate(cls, X, y, cv=tscv, scoring=scoring, return_train_score=True)
        
        scores = {}
        scores['train_r2'] = []
        scores['test_r2'] = []
        scores['train_neg_mean_absolute_error'] = []
        scores['test_neg_mean_absolute_error'] = []
        scores['train_neg_mean_squared_error'] = []
        scores['test_neg_mean_squared_error'] = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            #print(X_train.shape, X_train.index.min(), X_train.index.max())
            
            cls.fit(X_train, y_train)
            
            y_pred = cls.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            mse  = mean_squared_error(y_test, y_pred) 
            #acc  = accuracy_score(y_test, y_pred)

            y_pred_tr = cls.predict(X_train)
            tr_r2   = r2_score(y_train, y_pred_tr)
            tr_mae  = mean_absolute_error(y_train, y_pred_tr)
            tr_mse  = mean_squared_error(y_train, y_pred_tr) 

            scores['train_r2'].append(tr_r2)
            scores['test_r2'].append(r2)
            scores['train_neg_mean_absolute_error'].append(tr_mae)
            scores['test_neg_mean_absolute_error'].append(mae)
            scores['train_neg_mean_squared_error'].append(tr_mse)
            scores['test_neg_mean_squared_error'].append(mse)

        results.append([
            model.__name__,
            f"{mean(scores['train_r2']):.3f}",
            f"{mean(scores['test_r2']):.3f}",
            f"{mean(scores['train_neg_mean_absolute_error']):.3f}",
            f"{mean(scores['test_neg_mean_absolute_error']):.3f}",
            f"{mean(scores['train_neg_mean_squared_error']):.3f}",
            f"{mean(scores['test_neg_mean_squared_error']):.3f}",

        ])
    results = sorted(results, key=itemgetter(2), reverse=True) # sort table by test_r2  
    return results


def model_using_grid_search_cv_with_timeseriessplit(X, y):
    parameters = {
        "RandomForestRegressor" : { 
            'n_estimators': [20, 50, 100],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [i for i in range(5,15)]
        },
        "XGBRegressor" : {
            'nthread':[4], #when use hyperthread, xgboost may become slower
            'objective':['reg:linear'],
            'learning_rate': [.03, 0.05, .07], #so called `eta` value
            'max_depth': [5, 6, 7,9, 10],
            'min_child_weight': [4],
            'silent': [1],
            'subsample': [0.7],
            'colsample_bytree': [0.7],
            'n_estimators': [500]
        },
        "LGBMRegressor" : {
            'num_leaves': [7, 14, 21, 28, 31, 50],
            'learning_rate': [0.1, 0.03, 0.003],
            'max_depth': [-1, 3, 5],
            'n_estimators': [50, 100, 200, 500],
        },
        "GradientBoostingRegressor" : {
            'learning_rate': [0.01,0.02,0.03,0.04],
            'subsample'    : [0.9, 0.5, 0.2, 0.1],
            'n_estimators' : [100,500,1000, 1500],
            'max_depth'    : [4,6,8,10]
        },
        "DecisionTreeRegressor" : {
            "criterion": ["mse", "mae"],
            "min_samples_split": [10, 20, 40],
            "max_depth": [2, 6, 8],
            "min_samples_leaf": [20, 40, 100],
            "max_leaf_nodes": [5, 20, 100],
        },
        "BayesianRidge" : {'alpha':[1, 10]}
    }
    # https://goldinlocks.github.io/Time-Series-Cross-Validation/ 
    results = []
    for model in [
        RandomForestRegressor, # https://pierpaolo28.github.io/blog/blog25/
        xgboost.XGBRegressor, # https://www.kaggle.com/code/jayatou/xgbregressor-with-gridsearchcv/script
        LGBMRegressor, # https://stackoverflow.com/questions/63356595/gridsearchcv-with-lgbmregressor-cant-find-best-parameters
        GradientBoostingRegressor, # https://www.projectpro.io/recipes/find-optimal-parameters-using-gridsearchcv-for-regression
        DecisionTreeRegressor, # https://www.kaggle.com/code/marklvl/decision-tree-regressor-on-bike-sharing-dataset/notebook
        #BayesianRidge # https://stackoverflow.com/questions/57376860/how-to-run-gridsearchcv-with-ridge-regression-in-sklearn
        ]:
        
        split = int(len(X) * 0.7)

        X_train = X[:split]
        y_train = y[:split]

        X_test = X[split:]
        y_test = y[split:]

        print("Training:", model.__name__)
        
        cls = model()
        
        cv_splits= 10
        tscv = TimeSeriesSplit(n_splits=cv_splits)
            
        xgb_grid = GridSearchCV(cls, parameters[model.__name__], cv=tscv, n_jobs=5, verbose=True)
        
        xgb_grid.fit(X_train, y_train)
        
        y_true = y_test.values
        y_pred = xgb_grid.predict(X_test)
        
        r2 = metrics.r2_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
        mse=metrics.mean_squared_error(y_true, y_pred) 
        
        results.append([
            model.__name__,
            f"{r2:.3f}",
            f"{mean_absolute_error:.3f}",
            f"{mse:.3f}",
        ])
    results = sorted(results, key=itemgetter(1), reverse=True) # sort table by r2   
    return results





















