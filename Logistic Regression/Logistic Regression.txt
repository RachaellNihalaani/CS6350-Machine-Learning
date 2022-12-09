Assignment 5

Question 

[Bonus] [30 points] We will implement the logistic regression model with stochastic
gradient descent. We will use the dataset “bank-note.zip” in Canvas. Set the maximum
number of epochs T to 100. Don’t forget to shuffle the training examples at the start of
each epoch. Use the curve of the objective function (along with the number of updates)
to diagnosis the convergence. We initialize all the model parameters with 0.

(a) [10 points] We will first obtain the MAP estimation. In order for that, we assume
each model parameter comes from a Gaussian prior distribution,
p(wi) = N(wi|0,v) = 1√2πv exp(− 1
2vw2i )
where v is the variance. From the paper problem 4, you should be able to write
down the objective function and derive the gradient. Try the prior variance v
from {0.01,0.1,0.5,1,3,5,10,100}. Use the schedule of learning rate: γt = γ0
1+γ0d t.
Please tune γ0 and d to ensure convergence. For each setting of variance, report
your training and test error.

(b) [5 points] We will then obtain the maximum likelihood (ML) estimation. That
is, we do not assume any prior over the model parameters, and just maximize the
logistic likelihood of the data. Use the same learning rate schedule. Tune γ0 and
d to ensure convergence. For each setting of variance, report your training and
test error.

(c) [3 points] How is the training and test performance of the MAP estimation
compared with the ML estimation? What can you conclude? What do you think
of v, as compared to the hyperparameter C in SVM?
