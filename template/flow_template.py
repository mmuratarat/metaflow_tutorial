import metaflow 
import pandas as pd
import numpy as np

class TemplateFlow(metaflow.FlowSpec):

    @metaflow.step
    def start(self):
        """
        This is the starting point of the DAG
        """

        print("Workflow is starting...")
        print(f"flow name: {metaflow.current.flow_name}")
        print(f"run ID: {metaflow.current.run_id}")
        print(f"username: {metaflow.current.username}")

        # Call next step in DAG with self.next(...)
        self.next(self.get_data)

    
    @metaflow.step
    def get_data(self):
        """
        This step reads in the data from a CSV file using pandas
        """
        print("Reading the data...")

        #TODO

        self.next(self.split_data)

    @metaflow.step
    def split_data(self):
        print("Splitting the data into two: training and testing...")
        """
        This step splits the data into training and testing sets
        """

        #TODO

        self.next(self.data_preprocessing)

    @metaflow.step
    def data_preprocessing(self):
        """
        This step scales the numeric columns and one-hot encodes the categorical columns for both training and test sets.
        """
        print("Processing the data...")

        #TODO

        self.next(self.tune_hyperparameters)

    @metaflow.step
    def tune_hyperparameters(self):
        """
        This step tunes the hyperparameters of the logistic regression model
        """
        print("Tuning the hyperparameters of the model using Grid Search with Cross Validation...")

        #TODO

        self.next(self.train_model)

    @metaflow.step
    def train_model(self):
        """
        This step trains a logistic regression model on the training data 
        with the best parameters obtained by using GridSearchCV
        """

        print("Training the model with the best parameters...")

        #TODO

        self.next(self.evaluate_model)

    @metaflow.step
    def evaluate_model(self):
        """
        This step evaluates the trained model on the test data and returns the accuracy
        """
        print("Evaluating the model on training and test sets and printing out the accuracy scores...")

        #TODO

        self.next(self.save_model)

    @metaflow.step
    def save_model(self):
        """
        This step saves the model with pickle
        """
        print("Saving the model")

        #TODO

        self.next(self.deploy)

    @metaflow.step
    def deploy(self):
        """
        This step deploys the model
        """
        print("you'll deploy your model")

        self.next(self.end)

    @metaflow.step
    def end(self):
        """
        This is the ending point of the DAG
        """
        print("Workflow is done!")

if __name__ == '__main__':
    TemplateFlow()    
