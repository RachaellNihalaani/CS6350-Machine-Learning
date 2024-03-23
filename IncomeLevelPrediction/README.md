# Income-Level-Prediction
The Income Level Prediction project is a classification problem with the prediction task to determine whether a person makes over 50K a year given the census information. This project has been developed for CS5350/6350 Machine Learning in Fall 2022 at the University of Utah.


## Description
### Task description
This dataset was extracted by by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the specific conditions. Prediction task is to determine whether a person makes over 50K a year given the census information. There are 14 attributes, including continuous, categorical and integer types. Some attributes may have missing values, recorded as question marks.

### Relevant paper
Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996. (PDF)

## Evaluation
The evaluation is based on **Area Under ROC (AUC)** curve, which is a value between 0 and 1. The higher AUC, the better the predictive performance. Note that AUC is the most commonly used measure in ML practice. It considers the cases of all possible thresholds that are used for (binary) classification, and calculates the area of the (TPR, FPR) curve of using these thresholds (TPR and FPR stands for True Positive Rate and False Positive Rate, respectively) as an overall measure of the model performance. Therefore, AUC is not restricted to the accuracy of any single threshold (e.g., 0.5 or 0). It is a comprehensive evaluation. A detailed introduction of AUC is given here.

The file should contain a header and have the following format:
```
ID,Prediction
3,0.3
8,-0.2
10,0.55
...
```

## Dataset Description
**Target label** 
* income > 50K: binary
  
**Listing of attributes**
* age: continuous
* workclass: {Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked}
* fnlwgt: continuous
* education: {Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool}
* education-num: continuous.
* marital-status: {Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse}
* occupation: {Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces}
* relationship: {Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
* race: {White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black}
* sex: {Female, Male}
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: {United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands}

