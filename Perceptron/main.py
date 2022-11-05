'''
Perceptron for binary classification task - Bank Note authentication

Data in "bank-note folder" -> train.csv and test.csv
Data desc -
    We use 4 attributions (the first 4 columns)
        1. variance of Wavelet Transformed image (continuous)
        2. skewness of Wavelet Transformed image (continuous)
        3. curtosis of Wavelet Transformed image (continuous)
        4. entropy of image (continuous)
    The label is the last column: genuine(1) or forged(0)
'''

import pandas as pd
import numpy as np
import random


# Perceptron class
class Perceptron:
    def __init__(self):
        self.epochs = 10  # Set max epochs
        self.learning_rate = 0.01
        self.weights = None
        self.votes = None

    # Q2(a) Define Standard Perceptron
    def standard(self, X, y):
        # Add bias term of 1
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        # Initial attribute weight is set to 0
        self.weights = np.zeros(X.shape[1])

        # In each epoch
        for i in range(self.epochs):
            # shuffle - randomly select a subset of the population
            sub = random.sample(range(len(X)), len(X))
            # iterate through the subset
            for j in sub:
                # check condition if there is an error
                if y[j]*np.dot(self.weights, X[j]) <= 0:
                    # Update weight vector
                    self.weights += self.learning_rate * (y[j]*X[j])

    # Q2(b) Define Voted Perceptron
    def voted(self,X,y,):
        # Add bias term of 1
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        # Initial attribute weight is set to 0
        self.weights = [np.zeros(X.shape[1])]

        m = 0
        c = [0]  # Count of predictions made by w_m

        # In each epoch
        for i in range(self.epochs):
            for j in range(X.shape[0]):
                # check condition if there is an error
                if y[j]*np.dot(self.weights[m], X[j]) <= 0:
                    # update weight vector
                    self.weights.append(self.weights[m] + self.learning_rate * (y[j]*X[j]))
                    m += 1
                    c.append(1)
                else:
                    c[m] += 1

        # array of [distinct weight vectors, counts]
        self.votes = np.array(list(zip(self.weights, c)), dtype=object)

    # Q2(c) Define Average Perceptron
    def average(self, X, y,):
        # Add bias term of 1
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        # Initial attribute weight is set to 0
        self.weights = np.zeros(X.shape[1])
        a = np.zeros(X.shape[1])

        # In each epoch
        for i in range(self.epochs):
            for j in range(X.shape[0]):
                # check condition if there is an error
                if y[j]*np.dot(a, X[j]) <= 0:
                    # Update weight vector
                    a += self.learning_rate * (y[j]*X[j])
                self.weights += a

    # Prediction function for standard and average perceptron
    def predict(self, X):
        # Add bias term of 1
        X = np.append(np.ones((X.shape[0],1)), X, axis=1)
        sgn = lambda data: np.sign(np.dot(self.weights, data))
        y_pred = [sgn(data) for data in X]
        return y_pred

    # Prediction function for voted perceptron
    def predict2(self, X):
        # Add bias term of 1
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        y_pred = []
        for i in range(len(X)):
            sum_weights = 0
            for w, c in self.votes:
                sum_weights += c * np.sign(np.dot(w, X[i]))
            y_pred.append(np.sign(sum_weights))
        return y_pred

    def test_error(self, y, y_pred):
        return 1 - (np.sum(y == y_pred)/len(y))


if __name__ == "__main__":

    # Read train and test data; split into X and y
    train_data = pd.read_csv('./bank-note/train.csv', header=None)
    X_train = train_data.iloc[:, :4]
    y_train = train_data.iloc[:, 4]
    y_train_2 = np.array([-1 if i == 0 else 1 for i in y_train])  # for output to be {-1, 1}

    test_data = pd.read_csv('./bank-note/test.csv', header=None)
    X_test = test_data.iloc[:, :4]
    y_test = test_data.iloc[:, 4]
    actual_label = np.array([-1 if i ==0 else 1 for i in y_test])  # for output to be {-1, 1}

    # Q2(a) Implement Standard Perceptron
    standard_perceptron = Perceptron()
    standard_perceptron.standard(X_train, y_train_2)
    predicted_label = standard_perceptron.predict(X_test)
    print("For Standard Perceptron:")
    print("The learned weight vector is = ", standard_perceptron.weights)
    print("The average prediction error on the test dataset is = ", standard_perceptron.test_error(actual_label, predicted_label))

    # Q2(b) Implement Voted Perceptron
    voted_perceptron = Perceptron()
    voted_perceptron.voted(X_train, y_train_2)
    predicted_label = voted_perceptron.predict2(X_test)
    print("\n\nFor Voted Perceptron:")
    print("Distinct Weight Vectors \t\t\t\t\t\t\t\t\t\t\t Counts")
    for vect in voted_perceptron.votes:
        print(f"{vect[0]}\t\t\t{vect[1]}")
    print("The average prediction error on the test dataset is = ", voted_perceptron.test_error(actual_label, predicted_label))

    # Q2(c) Implement Average Perceptron
    average_perceptron = Perceptron()
    average_perceptron.average(X_train, y_train_2)
    predicted_label = average_perceptron.predict(X_test)
    print("\n\nFor Average Perceptron:")
    print("The learned weight vector is = ", average_perceptron.weights)
    print("The average prediction error on the test dataset is = ", average_perceptron.test_error(actual_label, predicted_label))