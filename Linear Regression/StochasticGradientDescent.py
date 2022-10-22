# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StochasticGradientDescent:
    # Constructor
    def __init__(self, r=0.001, t=0.05 ):
        self.learning_rate = r
        self.threshold = t
        self.epochs = 10000
        self.weights = None
        self.intercept = None
        self.cost = []

    # Fit model
    def fit(self, X, target):
        self.intercept = 0
        self.weights = np.ones(X.shape[1])

        # Initial update
        pred_output = self.predict(X)
        this_cost = StochasticGradientDescent.calc_cost(target, pred_output)
        update_iter = 0

        # Run algorithm till convergence condition is met
        while this_cost > self.threshold and update_iter < self.epochs:
            for i in range(X.shape[0]):
                # Take a random index
                index = np.random.randint(0, X.shape[0])

                # Calculate predicted value
                output = self.intercept + np.dot(X.iloc[index], self.weights)

                # Calculate gradient
                grad_wt = -2 * np.dot((target[index] - output), X.iloc[index])
                grad_int = -2 * (target[index] - output)

                # Update
                self.weights -= self.learning_rate * grad_wt
                self.intercept -= self.learning_rate * grad_int

            # Calculate predicted value
            pred_output = self.predict(X)

            # Calculate cost and append to array
            this_cost = StochasticGradientDescent.calc_cost(target , pred_output)
            self.cost.append(this_cost)

            update_iter += 1

    def calc_cost(target, predicted):
        return np.mean(np.square(target - predicted))

    def predict(self, X_test):
        return self.intercept + np.dot(X_test, self.weights)

if __name__ == '__main__':

    # Read and Split Training Data
    train_data = pd.read_csv('./Concrete/train.csv',
                             names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP'])
    X_train = train_data.iloc[:, :-1]
    y_train = train_data['SLUMP']

    # Read and Split Testing Data
    test_data = pd.read_csv('./Concrete/test.csv',
                            names=['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'SLUMP'])
    X_test = test_data.iloc[:, :-1]
    y_test = test_data['SLUMP']

    # Initialize variables
    learning_rates = [0.25, 0.125, 0.1, 0.05, 0.01, 0.001]
    cost = {}

    # implement model
    for r in learning_rates:
        # Create Instance object of Class
        st_grad_desc = StochasticGradientDescent()

        # Fit training data on model
        st_grad_desc.fit(X_train, y_train)

        # Print output
        print("At learning rate = ", r,
              ", \nWeight Vector is = ", st_grad_desc.weights,
              " \nIntercept is = ", st_grad_desc.intercept,
              " \nCost Function on test data is = ", StochasticGradientDescent.calc_cost(y_test, st_grad_desc.predict(X_test)), "\n")

        # Store cost corresponding to r
        cost[r] = st_grad_desc.cost

    # Plot figure - Cost Function vs Number of Iterations
    graph_color = ['pink', 'red', 'orange', 'yellow', 'blue', 'green']
    color_iter = 0

    # Iterate over cost functions
    for i in cost:
        iterations = [x for x in range(len(cost[i]))]

        # Plot figure
        plt.plot(iterations,cost[i],color = graph_color[color_iter],label=i)
        plt.title("Cost Function vs Number of Updates")
        plt.xlabel("Number of Updates")
        plt.xscale("log")
        plt.ylabel("Cost Function")
        plt.legend()

        color_iter += 1

    plt.show()
