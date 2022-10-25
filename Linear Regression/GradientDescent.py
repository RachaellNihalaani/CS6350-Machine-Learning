# Importing required libraries
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


class GradientDescent:
    # Constructor
    def __init__(self, r=0, t=0.0000001):
        self.learning_rate = r
        self.threshold = t
        self.weights = None
        self.intercept = None
        self.cost = []

    # Fit model
    def fit(self, X, target):
        self.weights = np.ones(X.shape[1], dtype='float64')
        self.intercept = 0

        norm_diff = norm(self.weights, 2)

        # Run algorithm till convergence condition is met i.e. till norm > threshold
        while norm_diff > self.threshold:

            # Calculate predicted value
            output = self.intercept + np.dot(X, self.weights) # + or - ???????????

            # Calculate cost and append to array
            this_cost = np.mean(np.square(target - output))
            self.cost.append(this_cost)

            # Calculate gradient
            grad_wt = -2 * np.dot((target - output), X) / X.shape[0]
            grad_int = -2 * np.mean(target - output)

            # Calculate norm of weight vector difference
            norm_diff = norm(((self.weights - self.learning_rate*grad_wt) - self.weights),2)

            # Update weight vector
            self.weights -= self.learning_rate * grad_wt
            self.intercept -= self.learning_rate * grad_int

    @staticmethod
    def calc_cost(target, predicted):
        return np.mean(np.square(target - predicted))

    def predict(self, X_test):
        return self.intercept + np.dot(X_test, self.weights)

if __name__ == '__main__':

    # Read and Split Training Data
    train_data = pd.read_csv('./Concrete/train.csv', names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP'])
    X_train = train_data.iloc[:, :-1]
    y_train = train_data['SLUMP']

    # Read and Split Testing Data
    test_data = pd.read_csv('./Concrete/test.csv', names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP'])
    X_test = test_data.iloc[:, :-1]
    y_test = test_data['SLUMP']

    # Initialize variables
    learning_rates = [0.25, 0.125, 0.1, 0.05, 0.01, 0.001]
    cost = {}

    # Fit Gradient Descent function and calculate cost function value - vary across different learning rates
    for r in learning_rates:
        # Create Instance object of Class
        grad_desc = GradientDescent(r)

        # Fit training data on model
        grad_desc.fit(X_train, y_train)

        # Print output
        print("At learning rate = ", r,
              ",\nWeight Vector is = ", grad_desc.weights,
              "\nIntercept is = ", grad_desc.intercept,
              "\nCost Function on test data is = ", GradientDescent.calc_cost(y_test, grad_desc.predict(X_test)), "\n")

        # Store cost corresponding to r
        cost[r] = grad_desc.cost

    # Plot figure - Cost Function vs Number of Iterations
    graph_color = ['pink', 'red', 'orange', 'yellow', 'blue', 'green']
    color_iter = 0

    # Iterate over cost functions
    for i in cost:
        iterations = [x+1 for x in range(len(cost[i]))]

        # Plot figure
        plt.plot(iterations, cost[i], color=graph_color[color_iter], label=i)
        plt.title("Cost Function vs Number of Iterations")
        plt.xlabel("Number of Iterations")
        plt.xscale("log")
        plt.ylabel("Cost Function")
        plt.legend()

        color_iter += 1

    plt.show()
    plt.savefig("out/gradient-output.png")
    plt.clf()
