# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Import our Logistic Regression file
from LogisticRegression import LogisticRegression


# function to load the training and testing data
def load_data(folder_path):
    train = folder_path + '/train.csv'
    test = folder_path + '/test.csv'
    train_X, train_y = load_csv(train)
    test_X, test_y = load_csv(test)

    # Create feature array and convert labels from {0,1} to {-1,1}
    train_X = np.array(train_X, dtype=float)
    test_X = np.array(test_X, dtype=float)
    train_y = convert_labels(train_y)
    test_y = convert_labels(test_y)

    return train_X, train_y, test_X, test_y


# function to split csv file into data and labels
def load_csv(file_name):
    data = []  # empty list

    # load csv as file in read mode
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    # Convert to numpy array
    all_data = np.array(data)

    # Data = first 4 columns
    data = all_data[:, :-1]
    # Label = last column
    labels = all_data[:, -1]

    return data, labels


# function to convert labels from {0,1} to {-1,1}
def convert_labels(labels):
    new_labels = [-1 if int(label) <= 0 else 1 for label in labels]
    return np.array(new_labels)


def make_graph_MAP(models, var, title):
    plt.figure(figsize=(10, 10), dpi=300)
    for idx, model in enumerate(models):
        plt.plot(np.arange(1, model.epochs + 1), model.train_errors,
                 label=f'Variance= {var[idx]}, Test Error= {model.test_error:.4f}')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Error')
    plt.legend()
    plt.title(title)
    plt.savefig(f'output/{title}.png')
    plt.clf()

    for idx, model in enumerate(models):
        plt.plot(np.arange(1, len(model.train_loss) + 1), model.train_loss)
        plt.title(f'LR MAP Obj Fn for Variance = {model.variance}')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Objective Function Value')
        plt.savefig(f'output/LR MAP Obj Fn for Variance = {model.variance}.png')
        plt.clf()


def make_graph_ML(models, var, title):
    plt.figure(figsize=(10, 10), dpi=300)
    for idx, model in enumerate(models):
        plt.plot(np.arange(1, model.epochs + 1), model.train_errors,
                 label=f'Variance= {var[idx]}, Test Error= {model.test_error:.4f}')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Error')
    plt.legend()
    plt.title(title)
    plt.savefig(f'output/{title}.png')
    plt.clf()

    for idx, model in enumerate(models):
        plt.plot(np.arange(1, len(model.train_loss) + 1), model.train_loss)
        plt.title(f'LR ML Obj Fn for Variance = {model.variance}')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Objective Function Value')
        plt.savefig(f'output/LR ML Obj Function for Variance = {model.variance}.png')
        plt.clf()


def LR():
    train_X, train_y, test_X, test_y = load_data('./data/bank-note')

    prior_variance_values = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    # MAP Estimation
    print("Logistic Regression\n\nMaximum A Posteriori (MAP) Estimation")

    variance_models_MAP = []
    no_of_epochs = 100
    gamma_0 = 0.0001
    d = 100

    print(f'After {no_of_epochs} epochs,')
    for v in prior_variance_values:
        print(f'For Variance = {v}:')

        # Fit and train model
        lr_map = LogisticRegression(train_X=train_X, train_y=train_y, gamma_0=gamma_0, d=d, mode='MAP', variance=v, epochs=no_of_epochs)
        lr_map.train()

        # Compute and save Training and Testing errors
        train_error = lr_map.calc_error(train_X, train_y)
        test_error = lr_map.calc_error(test_X, test_y)
        lr_map.save_errors(train_error, test_error)

        # Append the trained model to array (for plots)
        variance_models_MAP.append(lr_map)

        # Print output
        print(f'Training Error: {train_error:.4f}, Test Error: {test_error:.4f}\n')

    # Plot MAP
    make_graph_MAP(variance_models_MAP, prior_variance_values,
                  f'LR MAP Training Errors for Epochs={no_of_epochs}, Gamma_0={gamma_0} and d={d}')

    # ML Estimation
    print("\n\nMaximum Likelihood (ML) Estimation")

    variance_models_ML = []
    no_of_epochs = 100
    gamma_0 = 0.0001
    d = 100

    print(f'After {no_of_epochs} Epochs,')
    for v in prior_variance_values:
        print(f'For Variance = {v}:')
        # Fit and train model
        lr_ml = LogisticRegression(train_X=train_X, train_y=train_y, gamma_0=gamma_0, d=d, mode='ML', variance=v, epochs=no_of_epochs)
        lr_ml.train()

        # Compute and save Training and Testing errors
        train_error = lr_ml.calc_error(train_X, train_y)
        test_error = lr_ml.calc_error(test_X, test_y)
        lr_ml.save_errors(train_error, test_error)

        # Append the trained model to array (for plots)
        variance_models_ML.append(lr_ml)

        # Print output
        print(f'Training Error: {train_error:.4f}, Testing Error: {test_error:.4f}')

    # Plot graphs
    make_graph_ML(variance_models_ML, prior_variance_values,
                  f'LR ML Training Errors for Epochs={no_of_epochs}, Gamma_0={gamma_0} and d={d}')


if __name__ == "__main__":
    LR()
