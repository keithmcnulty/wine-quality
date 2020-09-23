import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

## general training prep nodes

def split_data(df: pd.DataFrame, parameters: dict) -> list:
    """
    Split and select data for modeling
    :param df: Pandas Dataframe
    :param parameters: split paramaters
    :return: Pandas Dataframe
    """

    X = df[parameters["input_cols"]]
    y = df[parameters["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])

    return [X_train, X_test, y_train, y_test]

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    """
    Scale data for modelling
    :param X_train: Pandas DataFrame
    :param X_test: Pandas DataFrame
    :return: List of Pandas DataFrames
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    return [X_train_scaled, X_test_scaled]

## Logistic Regression nodes

from sklearn.linear_model import LogisticRegression


def train_logit(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    """
    Train Logistic Regress model
    :param X_train: Pandas DataFrame
    :param y_train: Pandas DataFrame
    :return: object of class LinearRegression
    """
    logit = LogisticRegression(max_iter=10000)
    logit.fit(X_train, y_train.values.ravel())
    return logit


## KNN Crossvaidated Classifier

from sklearn.neighbors import KNeighborsClassifier

def train_knn_crossvalidated(
        X_train: pd.DataFrame, y_train:pd.DataFrame, parameters: dict
) -> KNeighborsClassifier:
    """
    Train crossvalidated KNN Classifier
    :param X_train: Pandas DataFrame
    :param y_train: Pandas DataFrame
    :param parameters: Parameters for cross validation
    :return: Model of class KNNClassifier
    """

    param_dist = {'n_neighbors': parameters['knn_n_neighbors']}
    kfold = KFold(n_splits=parameters['k'],
                  shuffle=parameters['k-shuffle'],
                  random_state=parameters['random_state'])

    knnmodel = KNeighborsClassifier()

    knn_clf = RandomizedSearchCV(knnmodel, param_distributions=param_dist,
                                 n_iter=parameters['n_iter'], scoring=parameters['scoring'],
                                 error_score=parameters['error_score'], verbose=parameters['verbose'],
                                 n_jobs=parameters['n_jobs'], cv=kfold, random_state=parameters['random_state'])

    knn_clf.fit(X_train, y_train.values.ravel())
    return knn_clf

## XGBoost classifier

from xgboost import XGBClassifier

def train_xgb_crossvalidated(
        X_train: pd.DataFrame, y_train:pd.DataFrame, parameters: dict
) -> XGBClassifier:
    """
    Train crossvalidated XGB Classifier
    :param X_train: Pandas DataFrame
    :param y_train: Pandas DataFrame
    :param parameters: Parameters for cross validation
    :return: Model of class KNNClassifier
    """

    param_dist = {'n_estimators': stats.randint(1, 100),
                  'learning_rate': stats.uniform(0.01, 0.6),
                  'subsample': parameters['subsample'],
                  'max_depth': parameters['xgb_max_depth'],
                  'colsample_bytree': parameters['colsample_bytree'],
                  'min_child_weight': parameters['xgb_min_child_weight']
                  }

    kfold = KFold(n_splits=parameters['k'],
                  shuffle=parameters['k-shuffle'],
                  random_state=parameters['random_state'])

    xgbmodel = XGBClassifier()

    xgb_clf = RandomizedSearchCV(xgbmodel, param_distributions=param_dist,
                                 n_iter=parameters['n_iter'], scoring=parameters['scoring'],
                                 error_score=parameters['error_score'], verbose=parameters['verbose'],
                                 n_jobs=parameters['n_jobs'], cv=kfold, random_state=parameters['random_state'])

    xgb_clf.fit(X_train, y_train.values.ravel())
    return xgb_clf





## General model output nodes

def generate_classification_report(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Generate classification report for model
    :param model: model object
    :param X_test: Pandas DataFrame
    :param y_test: Pandas Dataframe of test target values
    :return: Classification Report
    """
    y_pred = model.predict(X_test)
    logger = logging.getLogger(__name__)
    logger.info(classification_report(y_test, y_pred))