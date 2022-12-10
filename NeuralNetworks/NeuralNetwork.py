# Import Libraries
import numpy as np


class NeuralNetwork():
    def __init__(self, num_input_nodes, num_output_nodes, hidden_layer_widths, epochs, gamma_0, d,
                 weight_initialization='gaussian', verbose=False):
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.hidden_layer_widths = hidden_layer_widths
        self.epochs = epochs
        self.gamma_0 = gamma_0
        self.d = d
        self.train_error_ar = []
        self.test_error = 100
        self.train_error = 100
        self.model_weights = self.initialize_weights(mode=weight_initialization)
        self.verbose = verbose

    def train(self, train_X, train_y):
        train_errors_ar = []
        for iteration in range(1, self.epochs + 1):

            # Shuffle data
            rand_i = np.arange(train_X.shape[0])
            np.random.shuffle(rand_i)
            train_X_random = train_X[rand_i]
            train_y_random = train_y[rand_i]

            for x, y in zip(train_X_random, train_y_random):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_learning_rate(t=iteration)
                self.update_network_parameters(changes_to_w)

            train_error = self.calc_error(train_X, train_y)
            train_accuracy = 1 - train_error
            if self.verbose:
                print(f'Epoch: {iteration}, Train Accuracy: {train_accuracy:.3f}, Train Error: {train_error:.3f}')
            train_errors_ar.append(train_error)

        return train_errors_ar

    def update_learning_rate(self, t):
        self.learning_rate = self.gamma_0 / (1 + ((self.gamma_0 / self.d) * t))

    def initialize_weights(self, mode):
        # number of nodes in each layer
        input_layer = self.num_input_nodes
        hidden_1 = self.hidden_layer_widths
        hidden_2 = self.hidden_layer_widths
        output_layer = self.num_output_nodes
        weights = {}
        if mode == 'gaussian':
            weights = {
                'W1': np.random.normal(loc=0, scale=1, size=(hidden_1, input_layer)),
                'W2': np.random.normal(loc=0, scale=1, size=(hidden_2, hidden_1)),
                'W3': np.random.normal(loc=0, scale=1, size=(output_layer, hidden_2))
            }
        elif mode == 'zero':
            weights = {
                'W1': np.zeros((hidden_1, input_layer)),
                'W2': np.zeros((hidden_2, hidden_1)),
                'W3': np.zeros((output_layer, hidden_2))
            }
        return weights

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, x_train):
        params = self.model_weights

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.sigmoid(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        params = self.model_weights
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.sigmoid(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])


        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.model_weights[key] = self.model_weights[key] - self.learning_rate * value

    def calc_error(self, data, labels):
        predictions = []
        for x, y in zip(data, labels):
            output = self.forward_pass(x)
            pred = 1 if output >= 0.5 else -1
            predictions.append(pred == int(y))
        return 1 - np.mean(predictions)

    def save_error(self, train_error_ar,  train_error, test_error ):
        self.train_error_ar = train_error_ar
        self.test_error  = test_error
        self.train_error  = train_error
