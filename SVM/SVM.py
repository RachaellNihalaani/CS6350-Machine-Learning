import numpy as np
from scipy.optimize import minimize


class PrimalSVM:
    '''
    SVM in primal domain with stochastic sub-gradient descent
    - set max epochs T to 100
    - shuffle training egs at start of each epoch
    - use curve of obj func to diagnose convergence
    - hyperparameter C = {100/873, 500/873, 700/573}
    '''
    def __init__(self, C, learning_rate_schedule, nepochs, train_data, train_labels, gamma_0=0.00001, a=10):
        '''
        :param C: Hyperparameter {00/873, 500/873, 700/573}
        :param learning_rate_schedule:
        :param nepochs:
        :param train_data:
        :param train_labels:
        :param gamma_0:
        :param a:
        '''
        self.C = C
        self.train_data = train_data
        self.train_labels = train_labels

        self.gamma_0 = gamma_0
        self.a = 10
        self.epoch_nums = nepochs
        if learning_rate_schedule == 'A':
            self.learning_rate_schedule = self.schedule_a
        else:
            self.learning_rate_schedule = self.schedule_b
        self.weights = np.full((self.train_data.shape[1] + 1), 0)
        self.train_error = None
        self.test_error = None
        self.objective_ar = []

    # schedule given in Q2a
    def schedule_a(self, t):
        return self.gamma_0 / (1 + ((self.gamma_0 / self.a) * t))

    # schedule given in Q2b
    def schedule_b(self, t):
        return self.gamma_0 / (1 * t)
    
    def train(self, trace_objective=False):
        self.weights = np.full((self.train_data.shape[1] + 1), 0)
        inputs = np.ones((self.train_data.shape[0], self.train_data.shape[1]+1))
        inputs[:,:-1] = self.train_data
        n = inputs.shape[0]
        for epoch in range(1,self.epoch_nums + 1):
            gamma_t = self.learning_rate_schedule(epoch)
            # Shuffling at start of each epoch
            shuffled_indices = np.arange(inputs.shape[0])
            np.random.shuffle(shuffled_indices)
            X = inputs[shuffled_indices]
            Y = self.train_labels[shuffled_indices]
            if trace_objective:
                objective_val = 0.5 * np.dot(self.weights[0:self.train_data.shape[1]], self.weights[0:self.train_data.shape[1]])
                for index in range(n):
                    x = X[index]
                    y = Y[index]
                    objective_val += self.C * max(0, 1 - y * np.dot(self.weights, x))
                self.objective_ar.append(objective_val)
            for index in range(n):
                x = X[index]
                y = Y[index]
                self.update_weights(gamma_t, n, x, y)

    def update_weights(self, lr, n, x, y):  
        val = (y * np.dot(x, self.weights))
        if val <= 1:
            self.weights = self.weights - lr * self.weights + lr * self.C * n * y * x
        else:
            self.weights = (1 - lr) * self.weights

    def evaluate(self, data, labels):
        inputs = np.ones((data.shape[0], data.shape[1]+1))
        inputs[:,:-1] = data
        wrong = 0
        for index in range(inputs.shape[0]):
            if labels[index] != self.predict(inputs[index]):
                wrong += 1
        error = ((wrong/inputs.shape[0])*100)
        return error
    
    def predict(self, x):
        return np.sign(np.dot(a=self.weights, b=x))

    def store_errors(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error
    

class DualSVM:
    def __init__(self, C, kernel_type, train_data, train_labels, gamma=0.1):
        self.C = C
        if kernel_type == 'TRIVIAL':
            self.kernel = self.identitykernel
        elif kernel_type == 'GAUSSIAN':
            self.kernel = self.gaussiankernel
        self.train_data = train_data.astype(float)
        self.train_labels = train_labels
        self.gamma = gamma
    
        self.N = self.train_data.shape[0]
        self.gram_matrix = self.compute_gram_matrix()
        self.psd_matrix = self.train_labels * self.gram_matrix * self.train_labels[:, np.newaxis]
        self.A = np.vstack((-np.eye(self.N), np.eye(self.N)))
        self.num_support_vectors = 0

        # Model Parameters
        self.a = None
        self.b = np.concatenate((np.zeros(self.N), self.C * np.ones(self.N)))
        self.weights = None
        self.train_error = None
        self.test_error = None
        self.support_vector_indices = None

    def identitykernel(self, x, y):
        # Dot product
        return np.dot(x,y)

    def gaussiankernel(self, i, j):
        # Gaussian Kernel Formula: k(i,j) = exp(- (||i-j||)^2 / gamma)
        return np.exp(-np.sum(np.square(i - j)) / self.gamma)

    def compute_gram_matrix(self):
        gram = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                gram[i, j] = self.kernel(self.train_data[i], self.train_data[j])
        return gram

    def train(self):
        def loss(a):
            return -(a.sum() - 0.5 * np.dot(a.T, np.dot(self.psd_matrix, a)))

        def jacobian_of_loss(a):
            return np.dot(a.T, self.psd_matrix) - np.ones_like(a)

        constraints = ({'type': 'ineq', 'fun': lambda x: self.b - np.dot(self.A, x), 'jac': lambda x: -self.A},
                    {'type': 'eq', 'fun': lambda x: np.dot(x, self.train_labels), 'jac': lambda x: self.train_labels})

        a_0 = np.random.rand(self.N)  
        opt_val = minimize(loss, a_0, jac=jacobian_of_loss, constraints=constraints, method='SLSQP', options={})

        self.a = opt_val.x  # optimal Lagrange multipliers
        self.a[np.isclose(self.a, 0)] = 0  
        self.a[np.isclose(self.a, self.C)] = self.C  

        support_indices = np.where(0 < self.a)[0] 
        margin_indices = np.where((0 < self.a) & (self.a < self.C))[0] 

        self.num_support_vectors = len(support_indices)
        self.support_vector_indices = support_indices

        a_x_label = self.a * self.train_labels
        cum_b = 0
        for j in margin_indices:
            x_j = self.train_data[j]
            kernel_eval = np.array([self.kernel(x_m, self.train_data[j]) if a_m > 0 else 0 for x_m, a_m in zip(self.train_data, self.a)])
            b = self.train_labels[j] - a_x_label.dot(kernel_eval)
            cum_b += b
        b = cum_b / len(margin_indices)
        
        self.b = b
        self.weights = np.sum(self.a[support_indices, None] * self.train_labels[support_indices, None] * self.train_data[support_indices], axis = 0)
    
    def evaluate(self, data, labels):
        data = data.astype(float)
        a_x_t = self.a * self.train_labels
        predictions = np.empty(len(data)) 
        for i, s in enumerate(data): 
            kernel_eval = np.array([self.kernel(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(self.train_data, self.a)])
            predictions[i] = a_x_t.dot(kernel_eval) + self.b
        num_wrong = np.sum((predictions * labels) < 0)
        error = (num_wrong / data.shape[0])
        return error

    def store_errors(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error
    

class DualKernelPerceptron:
    def __init__(self, X_train, y_train, gamma, epochs):
        self.X_train = X_train.astype(float)
        self.y_train = y_train
        self.gamma = gamma
        self.n = epochs
        self.a = np.zeros(self.X_train.shape[0])
        self.kernel = self.gaussiankernel
        self.train_error = None
        self.test_error = None

    def gaussiankernel(self, i, j):
        # Gaussian Kernel Formula: k(i,j) = exp(- (||i-j||)^2 / gamma)
        return np.exp(-np.sum(np.square(i - j)) / self.gamma)
    
    def train(self):   
        errors = []

        # Train till model converges, for 10 epochs
        for i in range(self.n):
            err = 0

            for j in range(self.a.size):
                # For misclassified examples
                if self.predict(self.X_train[j, :]) != self.y_train[j]:
                    err += 1
                    self.a[j] += 1
            errors.append(err)

            # check for model convergence
            if err == 0:
                break

    def calc_error(self, X, y_target):
        X = X.astype(float)
        misclassified = 0

        for i in range(X.shape[0]):
            if self.predict(X[i, :]) != y_target[i]:
                misclassified += 1
        error = misclassified / X.shape[0]
        return error

    def predict(self, x_i):
        # Prediction Function : sgn(sum(a_i * y_i * K(x_i, x_j))
        total = 0
        for i in range(self.a.size):
            total += self.a[i] * self.y_train[i] * self.kernel(self.X_train[i, :], x_i)
        prediction = np.where(total >= 0.0, 1, -1)
        return prediction

    def store_errors(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error

