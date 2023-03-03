import metaflow 
import pandas as pd
import numpy as np
from sklearn import set_config
set_config(transform_output = "pandas")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

class LogisticWorkFlow(metaflow.FlowSpec):

    @metaflow.step
    def start(self):
        """
        This is the starting point of the DAG
        """

        print("Workflow is starting...")
        print(f"flow name: {metaflow.current.flow_name}")
        print(f"run ID: {metaflow.current.run_id}")
        print(f"username: {metaflow.current.username}")

        self.next(self.get_data)

    
    @metaflow.step
    def get_data(self):
        """
        This step reads in the data from a CSV file using pandas
        """
        print("Reading the data...")
        cols_ = ['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased']
        datatypes_ = {'User ID': 'int64', 'Gender': 'object', 'Age': 'int64', 
                      'EstimatedSalary': 'float64', 'Purchased': 'int64'}
        
        self. data = pd.read_csv("Social_Network_Ads.csv", index_col='User ID', sep = ',', header='infer', dtype = datatypes_)
        self.next(self.split_data)

    @metaflow.step
    def split_data(self):
        print("Splitting the data into two: training and testing...")
        """
        This step splits the data into training and testing sets
        """
        X = self.data.drop('Purchased', axis=1)
        y = self.data['Purchased']

        self.numeric_columns = list(X.select_dtypes(include = np.number).columns)
        self.categorical_columns = list(X.select_dtypes(include = 'object').columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.next(self.data_preprocessing)

    @metaflow.step
    def data_preprocessing(self):
        """
        This step scales the numeric columns and one-hot encodes the categorical columns for both training and test sets.
        """
        print("Processing the data...")
        scaler_ = StandardScaler(with_mean = True, with_std = True)
        encoder_ = OneHotEncoder(categories = 'auto', drop = None, sparse_output = False, handle_unknown = 'error')
        
        scaler_.fit(self.X_train[self.numeric_columns])
        encoder_.fit(self.X_train[self.categorical_columns])

        self.X_train = pd.concat([scaler_.transform(self.X_train[self.numeric_columns]), encoder_.transform(self.X_train[self.categorical_columns])], axis = 1)
        self.X_test = pd.concat([scaler_.transform(self.X_test[self.numeric_columns]), encoder_.transform(self.X_test[self.categorical_columns])], axis = 1)
        self.next(self.tune_hyperparameters)

    @metaflow.step
    def tune_hyperparameters(self):
        """
        This step tunes the hyperparameters of the logistic regression model
        """
        print("Tuning the hyperparameters of the model using Grid Search with Cross Validation...")
        model = LogisticRegression()
        param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        search = GridSearchCV(model, param_grid, cv=10, refit=True, scoring='accuracy')
        search.fit(self.X_train, self.y_train)
        self.best_params = search.best_params_
        self.next(self.train_model)

    @metaflow.step
    def train_model(self):
        """
        This step trains a logistic regression model on the training data 
        with the best parameters obtained by using GridSearchCV
        """
        print("Training the model with the best parameters...")
        self.model = LogisticRegression(**self.best_params)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.evaluate_model)

    @metaflow.step
    def evaluate_model(self):
        """
        This step evaluates the trained model on the test data and returns the accuracy
        """
        print("Evaluating the model on training and test sets and printing out the accuracy scores...")
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        print(f"The model accuracy on training data with the best parameters is: {accuracy_score(self.y_train, y_train_pred)}")
        print(f"The model accuracy on test data with the best parameters is: {accuracy_score(self.y_test, y_test_pred)}")
        self.next(self.save_model)

    @metaflow.step
    def save_model(self):
        """
        This step saves the model with pickle
        """
        print("Saving the model")
        filename = 'finalized_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        self.next(self.end)

    @metaflow.step
    def end(self):
        """
        This is the ending point of the DAG
        """
        print("Workflow is done!")

if __name__ == '__main__':
    LogisticWorkFlow()    
