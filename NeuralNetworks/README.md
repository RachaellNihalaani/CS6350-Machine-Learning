Assignment 5

Question

[58 points] Now let us implement a three-layer artificial neural network for classifi-
cation. We will use the dataset, “bank-note.zip” in Canvas. The features and labels
are listed in the file “classification/data-desc.txt”. The training data are stored in the
file “classification/train.csv”, consisting of 872 examples. The test data are stored in
“classification/test.csv”, and comprise of 500 examples. In both the training and test
datasets, feature values and labels are separated by commas. The architecture of the
neural network resembles Figure 1, but we allow an arbitrary number of units in hidden
layers (Layer 1 and 2). So please ensure your implementation has such flexibility. We
will use the sigmoid activation function.

(a) [25 points] Please implement the back-propagation algorithm to compute the
gradient with respect to all the edge weights given one training example. For
debugging, you can use the paper problem 3 and verify if your algorithm returns
the same derivatives as you manually did.

(b) [17 points] Implement the stochastic gradient descent algorithm to learn the neu-
ral netowrk from the training data. Use the schedule of learning rate: γt = γ0
1+γ0d t.
Initialize the edge weights with random numbers generated from the standard
Gaussian distribution. We restrict the width, i.e., the number of nodes, of
each hidden layer (i.e., Layer 1 & 2 ) to be identical. Vary the width from
{5,10,25,50,100}. Please tune γ0 and d to ensure convergence. Use the curve
of the objective function (along with the number of updates) to diagnosis the
convergence. Don’t forget to shuffle the training examples at the start of each
epoch. Report the training and test error for each setting of the width.

(c) [10 points]. Now initialize all the weights with 0, and run your training algorithm
again. What is your training and test error? What do you observe and conclude?

(d) [6 points]. As compared with the performance of SVM (and the logistic regression
4
you chose to implement it; see Problem 3), what do you conclude (empirically)
about the neural network?

(e) [Bonus] [30 points] Please use PyTorch (or TensorFlow if you want) to fulfill
the neural network training and prediction. Please try two activation functions,
“tanh” and “RELU”. For “tanh”, please use the “Xavier’ initialization; and for
“RELU”, please use the “he” initialization. You can implement these initializa-
tions by yourselves or use PyTorch (or TensorFlow) library. Vary the depth from
{3,5,9} and width from {5,10,25,50,100}. Pleas use the Adam optimizer for
training. The default settings of Adam should be sufficient (e.g., initial learning
rate is set to 10−3). Report the training and test error with each (depth, width)
combination. What do you observe and conclude? Note that, we won’t provide
any link or manual for you to work on this bonus problem. It is YOUR JOB
to search the documentation, find code snippets, test, and debug with PyTorch
(or TensorFlow) to ensure the correct usage. This is what all machine learning
practitioners do in practice.
