# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Import Neural Network class
from NeuralNetwork import NeuralNetwork


def load_csv(file_name):
    X = []
    with open(file_name, 'r') as f:
        for line in f:
            X.append(line.strip().split(','))
    data = np.array(X)
    X = data[:, :-1]
    X = np.array(X, dtype=float)

    y = data[:, -1]
    y = [-1 if int(label) <= 0 else 1 for label in y]
    y = (np.array(y))

    return X, y


# Function to load data
def load_data():
    train_X, train_y = load_csv('./data/bank-note/train.csv')
    test_X, test_y = load_csv('./data/bank-note/test.csv')

    return train_X, train_y, test_X, test_y


def make_graph(models, widths, title):
    plt.figure(figsize=(10, 10), dpi=300)

    for idx, model in enumerate(models):
        plt.plot(np.arange(1, model.epochs + 1), model.train_error_ar,
                 label=f'Hidden Layer Width: {widths[idx]}(Test Loss: {model.test_error:.4f})')
    plt.legend()
    plt.title(title)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(f'./Outputs/{title}.png')
    plt.clf()


def NeuralNet():

    train_X, train_y, test_X, test_y = load_data()

    print("Backpropagation and Weights initialized with Standard Gaussian Distribution")
    models_ar = []
    num_epochs = 200
    gamma_0 = 0.0001
    d = 100

    print(f'At end of {num_epochs} Epochs, ')

    for hidden_layer_width in [5, 10, 25, 50, 100]:
        print(f'Width of Hidden layer = {hidden_layer_width}')

        # Build and train model
        nn_gau = NeuralNetwork(num_input_nodes=4, num_output_nodes=1, hidden_layer_widths=hidden_layer_width, epochs=num_epochs, gamma_0=gamma_0, d=d, verbose=False, weight_initialization='gaussian')
        train_errors_ar = nn_gau.train(train_X=train_X, train_y=train_y)
        train_error = nn_gau.calc_error(train_X, train_y)
        test_error = nn_gau.calc_error(test_X, test_y)
        nn_gau.save_error(train_errors_ar,  train_error, test_error)
        models_ar.append(nn_gau)

        print(f'Training Error= {train_error:.4f} and Testing Error= {test_error:.4f}')

    # Plot
    make_graph(models_ar, [5, 10, 25, 50, 100],
              f'Trainings Errors vs Epochs (Weights initialized with Gaussian) Dist_Epochs={num_epochs} Gamma_0={gamma_0} d={d}')

    print("Backpropagation and Weights initialized to Zero")
    models_ar = []
    num_epochs = 200
    gamma_0 = 0.0001
    d = 100

    print(f'At end of {num_epochs} Epochs, ')

    for hidden_layer_width in [5, 10, 25, 50, 100]:
        print(f'Width of Hidden layer = {hidden_layer_width}')
        nn_zero = NeuralNetwork(num_input_nodes=4, num_output_nodes=1, hidden_layer_widths=hidden_layer_width,  epochs=num_epochs, gamma_0=gamma_0, d=d, verbose=False, weight_initialization='zero')
        train_errors_ar = nn_zero.train(train_X=train_X, train_y=train_y)
        train_error = nn_zero.calc_error(train_X, train_y)
        test_error = nn_zero.calc_error(test_X, test_y)
        nn_zero.save_error(train_errors_ar, train_error, test_error)
        models_ar.append(nn_zero)

        print(f'Training Error= {train_error:.4f} and Testing Error= {test_error:.4f}')

    # Plot
    make_graph(models_ar, [5, 10, 25, 50, 100],
              f'Trainings Errors vs Epochs (for Weights initialized with Zero) Epochs={num_epochs} Gamma_0={gamma_0} d={d}')


if __name__ == "__main__":
    NeuralNet()
