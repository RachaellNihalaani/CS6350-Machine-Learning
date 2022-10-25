## Code Output:

For Single Decision Trees -   
Bias =  0.33962424  
Variance =  0.4213935744000002  
General Squared Error =  0.7610178144000002  
  
For Bagged Trees -   
Bias =  0.3512076  
Variance = 0.3105264768000003  
General Squared Error =  0.6617340768000003  


## Comparison of the Single Tree Learner and the Bagger Tree Learner:
Comparing the results of the two, we can conclude that the single decision tree learner tends to overfit, so it has a smaller bias but a larger variance. The bagged tree learner, on the other hand, has a similar bias but has a largely reduced variance, and a reduced general squared error as well.  
  
This difference in the values of bias and variance is caused by the fact that there is a bias-variance tradeoff. Also, Bagging algorithm averages a set of decision trees, which gives us these values of bias and variance and a better prediction in general.
