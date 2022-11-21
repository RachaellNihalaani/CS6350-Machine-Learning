import numpy as np
from SVM import PrimalSVM, DualSVM, DualKernelPerceptron
import matplotlib.pyplot as plt


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


# function to load the training and testing data
def load_data(folder_path):
    train = folder_path + '/train.csv'
    test = folder_path + '/test.csv'
    train_X, train_y = load_csv(train)
    test_X, test_y = load_csv(test)
    return train_X, train_y, test_X, test_y


# function to convert labels from {0,1} to {-1,1}
def convert_labels(labels):
    # new_labels = []
    # for label in labels:
    #     if int(label) <= 0:
    #         new_labels.append(-1)
    #     else:
    #         new_labels.append(1)
    new_labels = [-1 if int(label) <= 0 else 1 for label in labels]
    return np.array(new_labels)


# function to round
def round_ar(ar):
    ar_ = [(round(x, 4)) for x in ar]
    return ar_


# Execute Primal SVM Q2
def SVM_primal():
    print('Primal SVM')

    train_X, train_y, test_X, test_y = load_data('./data/bank-note')
    train_y = convert_labels(train_y)
    test_y = convert_labels(test_y)

    C = [100 / 873, 500 / 873, 700 / 873]
    verbose = True

    # Part A
    print('\nSchedule A of learning rate')
    svm_models_ar_a = []
    for c in C:
        # print('-------------------------------------------------------------------------')
        primal_svm_a = PrimalSVM(C=c,
                                 learning_rate_schedule='A',
                                 nepochs=100,
                                 train_data=train_X,
                                 train_labels=train_y,
                                 gamma_0=0.00001,
                                 a=100)
        primal_svm_a.train(trace_objective=True)
        train_error = primal_svm_a.evaluate(data=train_X, labels=train_y)
        test_error = primal_svm_a.evaluate(data=test_X, labels=test_y)
        primal_svm_a.store_errors(train_error, test_error)
        svm_models_ar_a.append(primal_svm_a)
        if verbose:
            print(f'For C = {c:.4f}\n'
                  f'Training Error = {train_error:.4f}\n'
                  f'Testing Error = {test_error:.4f}\n')

    # Part B
    print('\n\nSchedule B of learning rate')

    svm_models_ar_b = []
    for c in C:
        # print('-------------------------------------------------------------------------')
        primal_svm_b = PrimalSVM(C=c,
                                 learning_rate_schedule='B',
                                 nepochs=100,
                                 train_data=train_X,
                                 train_labels=train_y,
                                 gamma_0=0.00001)
        primal_svm_b.train(trace_objective=True)
        train_error = primal_svm_b.evaluate(data=train_X, labels=train_y)
        test_error = primal_svm_b.evaluate(data=test_X, labels=test_y)
        primal_svm_b.store_errors(train_error, test_error)
        svm_models_ar_b.append(primal_svm_b)
        if verbose:
            print(f'For C = {c:.4f}\n'
                  f'Training Error = {train_error}\n'
                  f'Test Error = {test_error}\n')

    # Part C -
    #  Comparing the difference of i) model parameters
    #                              ii) train errors
    #                              iii) test errors
    #  for each values of C
    print('\n\nDifferences between Schedules A and B')

    for idx, C_val in enumerate(C):
        # print('-------------------------------------------------------------------------')
        print(f'For C = {C_val:.4f}, ')
        diff_train = svm_models_ar_a[idx].train_error - svm_models_ar_b[idx].train_error
        diff_test = svm_models_ar_a[idx].test_error - svm_models_ar_b[idx].test_error

        print(
            f'Training Error : \n'
            f'Model A = {svm_models_ar_a[idx].train_error:.4f}\n'
            f'Model B = {svm_models_ar_b[idx].train_error:.4f}\n'
            f'Difference = {diff_train:.4f}\n')
        print(
            f'Testing Error : \n'
            f'Model A = {svm_models_ar_a[idx].test_error:.4f}\n'
            f'Model B = {svm_models_ar_b[idx].test_error:.4f}\n'
            f'Difference = {diff_test:.4f}\n')

        print(
            f'Weight Parameters : \n'
            f'Model A : \n'
            f'Size = {svm_models_ar_a[idx].weights.shape} \n'
            f'Weights : {round_ar(svm_models_ar_a[idx].weights)} \n'
            f'Model B : \n'
            f'Size = {svm_models_ar_b[idx].weights.shape} \n'
            f'Weights : {round_ar(svm_models_ar_b[idx].weights)} \n'
            f'Difference of Weights = {round_ar(svm_models_ar_a[idx].weights - svm_models_ar_b[idx].weights)}\n')

        # plotting the objective curve for comparison:
        plt.plot(svm_models_ar_a[idx].objective_ar, np.arange(1, svm_models_ar_a[idx].epoch_nums + 1), 'b',
                 label=f'C = {C_val:.4f} Model A')
        plt.plot(svm_models_ar_b[idx].objective_ar, np.arange(1, svm_models_ar_b[idx].epoch_nums + 1), 'y',
                 label=f'C = {C_val:.4f} Model B')
        plot_title = f'Objective Function Curve C = {C_val:.4f}'
        plt.title(plot_title)
        plt.xlabel('Number of Epochs')
        plt.ylabel('J')
        plt.legend()
        plt.savefig(f'./output/{plot_title}.png')
        plt.clf()


# Execute Dual SVM Q3abc
def SVM_dual():
    print('\n\nDual SVM')

    train_X, train_y, test_X, test_y = load_data('./data/bank-note')
    train_y = convert_labels(train_y)
    test_y = convert_labels(test_y)

    C = [100 / 873, 500 / 873, 700 / 873]

    verbose = True

    # Part A - Without Kernel
    print('Without Kernel\n')

    ds_output = []
    for c in C:
        # Calling the class from SVM.py (ds = dual SVM)
        ds = DualSVM(C=c, kernel_type='TRIVIAL', train_data=train_X, train_labels=train_y)
        ds.train()

        train_error = ds.evaluate(data=train_X, labels=train_y)
        test_error = ds.evaluate(data=test_X, labels=test_y)
        ds.store_errors(train_error, test_error)

        ds_output.append(ds)
        if verbose:
            print(f'For C = {c:.4f} :\n'
                  f'Training Error = {train_error:.4f}\n'
                  f'Testing Error = {test_error:.4f}\n'
                  f'Weight Vector w = {round_ar(ds.weights)}\n'
                  f'Bias b = {ds.b}\n')

    # Part B - With Gaussian Kernel
    print('\nWith Gaussian Kernel\n')

    ds_gk_output = []
    gamma_values = [0.1, 0.5, 1, 5, 100]

    for c in C:
        c_val_output = []
        for gamma in gamma_values:
            # Calling the class from SVM.py (ds_gk = dual SVM with gaussian kernel)
            ds_gk = DualSVM(C=c, kernel_type='GAUSSIAN', train_data=train_X, train_labels=train_y, gamma=gamma)
            ds_gk.train()

            train_error = ds_gk.evaluate(data=train_X, labels=train_y)
            test_error = ds_gk.evaluate(data=test_X, labels=test_y)
            ds_gk.store_errors(train_error, test_error)

            c_val_output.append(ds_gk)

            if verbose:
                print(
                    f'For C = {c:.4f} and Gamma = {gamma} :\n'
                    f'Training Error = {train_error:.4f}\n'
                    f'Testing Error = {test_error:.4f}\n'
                    f'Weight Vector w = {round_ar(ds_gk.weights)}\n'
                    f'Bias b = {ds_gk.b}\n')

        ds_gk_output.append(c_val_output)

    # Part C - (1) Number of Support Vectors for each C and Gamma (2) Number of overlapped Support Vectors between consecutive values of gamma for C=0.5727
    print('\nNumber of Support Vectors')

    for idx, c in enumerate(C):
        ds_gk_c_output = ds_gk_output[idx]
        print(f'For C = {c} and')

        for svm_model in ds_gk_c_output:
            print(f'Gamma = {svm_model.gamma}, '
                  f'Number of Support vectors = {svm_model.num_support_vectors}')

        print('\n')

    # for C = 500/873, model idx is 1
    print('\n\nNumber of Overlapped Support Vectors between consecutive gamma values for C = 500/873')
    svm_model = ds_gk_c_output[1]

    for i in range(0, len(svm_model) - 1):
        model_a = svm_model[i]
        model_b = svm_model[i + 1]

        num_overlapped_support_vectors = len(list(set(model_a.support_vector_indices).intersection(set(model_b.support_vector_indices))))

        print(f'For Gamma = {model_a.gamma} and {model_b.gamma}, '
              f'Number of Overlapped Support Vectors = {num_overlapped_support_vectors}')


# Execute Dual Kernel Perceptron SVM Q3d
def Perceptron_dual():
    print('\n\nDual Kernel Perceptron')

    train_X, train_y, test_X, test_y = load_data('./data/bank-note')
    train_y = convert_labels(train_y)
    test_y = convert_labels(test_y)

    verbose = True  # for detailed logging

    dkp_output = []
    gamma_values = [0.1, 0.5, 1, 5, 100]

    for g in gamma_values:
        # Calling the class from SVM.py (dkp = dual kernel perceptron)
        dkp = DualKernelPerceptron(train_data=train_X, train_labels=train_y, gamma=g)
        dkp.train()

        # Calculate error using evaluate function and store
        train_error = dkp.calc_error(data=train_X, labels=train_y)
        test_error = dkp.calc_error(data=test_X, labels=test_y)
        dkp.store_errors(train_error, test_error)

        dkp_output.append(dkp)

        if verbose:
            print(f'For Gamma=\t{g}, '
                  f'Training Error={train_error:.4f} and '
                  f'Testing Error={test_error:.4f}')


if __name__ == "__main__":
    SVM_primal()
    SVM_dual()
    Perceptron_dual()
