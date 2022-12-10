# Neural Networks using PyTorch

# Import Libraries
import numpy as np
import torch
from torch.nn import Module, Linear, Sequential
import torch.nn as nn

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# 'Xavier' initialization for tanh
def initialize_xavier(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.01)


# 'he' initialization for ReLU
def initialize_he(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.01)


class NeuralNet(Module):
    def __init__(self, layers, width, activation):
        super(NeuralNet, self).__init__()
        self.layers = layers
        self.width = width
        self.model = None
        self.flatten = nn.Flatten()
        self.activation = activation
        self.initialize_model_params()

    def initialize_model_params(self):
        if self.layers == 3:
            self.model = Sequential(
                Linear(4, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, 2)
            )
        elif self.layers == 5:
            self.model = Sequential(
                Linear(4, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, 2)
            )
        elif self.layers == 9:
            self.model = Sequential(
                Linear(4, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, self.width),
                self.activation,
                Linear(self.width, 2)
            )

        # Using 2 activation functions - tanh and ReLU
        if type(self.activation == nn.Tanh):
            # Using 'Xavier' initialization for tanh
            self.model.apply(initialize_xavier)
        elif type(self.activation) == nn.ReLU:
            # Using 'he' initialization for ReLU
            self.model.apply(initialize_he)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


# Functions
def load_csv(file_name):
    X = []
    with open(file_name, 'r') as f:
        for line in f:
            X.append(line.strip().split(','))
    data = np.array(X)
    X = data[:, :-1]
    y = data[:, -1]

    # Convert X to array and y to [0,1] labels
    X = np.array(X, dtype=float)
    y = [0 if int(i) == 0 else 1 for i in y]

    return X, y


# Function to load data
def load_data():
    train_data = './data/bank-note/train.csv'
    test_data = './data/bank-note/test.csv'

    train_X, train_y = load_csv(train_data)
    test_X, test_y = load_csv(test_data)

    return train_X, train_y, test_X, test_y


def train(model, optimizer, epochs, train_data, train_labels):
    model = model.to(device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        model_output = model(train_data)
        loss = torch.nn.functional.cross_entropy(model_output, train_labels)
        loss.backward()
        optimizer.step()


def calc_error(model, X, y):
    y_prob = torch.nn.Softmax(dim=1)(model(X))
    y_pred = y_prob.argmax(1)

    misclassified = 0
    for i in range(X.shape[0]):
        if y_pred[i] != y[i]:
            misclassified += 1
    err = misclassified / X.shape[0]
    return err


if __name__ == '__main__':
    print(f'Device: {device}')

    # Load data and convert to tensor
    train_X, train_y, test_X, test_y = load_data()

    # Convert data to tensor
    train_X_tensor = torch.tensor(train_X).to(device=device, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X).to(device=device, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y).to(device=device, dtype=torch.long).squeeze()
    test_y_tensor = torch.tensor(test_y).to(device=device, dtype=torch.long).squeeze()

    # Implementing 2 activation functions
    for activation_fun in ['ReLU', 'Tanh']:

        print(f'For Activation Function {activation_fun}')

        # Varying depth
        for depth in [3, 5, 9]:
            print(f'For Depth = {depth}')

            # Varying width
            for width in [5, 10, 25, 50, 100]:

                act_fn = torch.nn.ReLU() if activation_fun == 'ReLU' else torch.nn.Tanh()

                # Create and train model (using Adam)
                nn_pt = NeuralNet(layers=depth, width=width, activation=act_fn)
                train(model=nn_pt, optimizer=torch.optim.Adam(nn_pt.parameters(), lr=1e-3), epochs=100, train_data=train_X_tensor, train_labels=train_y_tensor)

                # Calculate error
                train_err = calc_error(nn_pt, train_X_tensor, train_y_tensor)
                test_err = calc_error(nn_pt, test_X_tensor, test_y_tensor)

                print(f'and Width = {width}, Training Error= {train_err:.4f} and Testing Error= {test_err:.4f}')
