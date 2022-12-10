# Import libraries
import numpy as np


# Get training data in random order - good practice
def get_rand_order(data, labels):
    new_order = np.arange(data.shape[0])
    np.random.shuffle(new_order)
    Data = data[new_order]
    Labels = labels[new_order]

    return Data, Labels


class LogisticRegression():
    def __init__(self, train_X, train_y, gamma_0, d, mode='MAP', variance = 0.1, epochs=100 ):

        self.train_y = train_y
        self.train_X = train_X

        self.weights = np.zeros(self.train_X.shape[1])  # Initialize weights to zeros
        self.epochs = epochs
        self.gamma_0 = gamma_0
        self.d = d
        self.variance = variance
        self.mode = mode

        self.train_loss = []
        self.train_errors = []
        self.test_error = 100
        self.train_error = 100

        self.verbose = False

    # Using given learning rate schedule
    def lr_sch(self, t):
        gamma_t = self.gamma_0 / (1 + ((self.gamma_0 / self.d) * t))
        return gamma_t

    # Function to calculate error
    def calc_error(self, data, y):
        pred_y = []
        for i in range(len(data)):
            pred_y_i = 1 if (np.inner(data[i, :], self.weights)) > 0 else -1
            pred_y.append(pred_y_i)

        misclassified = 0
        for idx in range(len(pred_y)):
            if pred_y[idx] != y[idx]:
                misclassified = misclassified + 1
        error = misclassified / len(pred_y)

        return error

    def save_errors(self, train_error, test_error):
        self.test_error = test_error
        self.train_error = train_error

    # Sigmoid Activation Function
    @staticmethod
    def sigmoid(x):
        sig = 0 if x < -100 else 1 / (1 + np.exp(-x))
        return sig

    # Function to calculate loss
    def calc_loss(self):
        loss_1 = 1 / (2 * self.variance) * np.inner(self.weights, self.weights)

        l2_arr = []
        for x, y in zip(self.train_X, self.train_y):
            p = -y * np.inner(self.weights, x)
            l2_arr.append(p if p > 100 else np.log(1 + np.exp(p)))
        loss_2 = np.sum(np.asarray(l2_arr))

        loss = loss_2 + loss_1
        return loss

    def train(self):
        if self.mode == 'MAP':
            self.MAP()
        elif self.mode == 'ML':
            self.ML()

    # Maximum A Posteriori (MAP) Estimation
    def MAP(self):

        iterations = 1

        for epoch in range(1, self.epochs + 1):
            new_X, new_y = get_rand_order(self.train_X, self.train_y)
            loss = []

            # For each row
            for x, y in zip(new_X, new_y):
                self.weights = np.asarray(self.weights)

                p = self.train_X.shape[0] * y * (1 - self.sigmoid(y * np.inner(self.weights, x)))
                sgd_val = np.asarray(
                    [self.weights[i] / self.variance - p * x[i] for i in range(self.train_X.shape[1])])
                self.weights = self.weights - self.lr_sch(iterations) * sgd_val
                loss.append(self.calc_loss())
                iterations = iterations + 1
            self.train_loss.extend(loss)

            # Calculate Training error and accuracy after each epoch
            train_error = self.calc_error(data=self.train_X, y=self.train_y)
            train_acc = 1 - train_error

            if self.verbose:
                print(f'For Epoch {epoch+1}, Train Error: {train_error:.4f}, Train Accuracy: {train_acc:.4f}')
            self.train_errors.append(train_error)

    # Maximum Liklihood (ML) Estimation
    def ML(self):

        iterations = 1

        for epoch in range(self.epochs):
            new_X, new_y = get_rand_order(self.train_X, self.train_y)
            loss = []

            # For each row
            for x, y in zip(new_X, new_y):
                self.weights = np.asarray(self.weights)
                p = self.train_X.shape[0] * y * (1 - self.sigmoid(y * np.inner(self.weights, x)))
                sgd_val = np.asarray([- p * x[i] for i in range(self.train_X.shape[1])])
                self.weights = self.weights - self.lr_sch(iterations) * sgd_val
                loss.append(self.calc_loss())
                iterations = iterations + 1
            self.train_loss.extend(loss)

            # Calculate Training error and accuracy after each epoch
            train_error = self.calc_error(data=self.train_X, y=self.train_y)
            train_acc = 1 - train_error

            if self.verbose:
                print(f'For Epoch {epoch+1}, Train Error: {train_error:.4f}, Train Accuracy: {train_acc:.4f}')
            self.train_errors.append(train_error)