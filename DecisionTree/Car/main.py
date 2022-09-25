#Decision Trees - ID3 Algorithm - Car Evaluation Task

#Importing Libraries
import numpy as np
import pandas as pd
import copy

#dataset columns and attributes
col_names = ['buying','maint','doors', 'persons', 'lug_boot', 'safety','label']
attributes = ['buying','maint','doors', 'persons', 'lug_boot', 'safety']
attribute_values = {'buying':['vhigh', 'high', 'med', 'low'], 'maint':['vhigh', 'high', 'med', 'low'],
    'doors':['2', '3', '4', '5more'], 'persons':['2','4','more'],
    'lug_boot':['small', 'med', 'big'],'safety':['low', 'med', 'high']}
label_values = ['unacc','acc','good','vgood']


#Reading datasets
train_data = pd.read_csv("train.csv",names=col_names)
test_data = pd.read_csv("test.csv",names = col_names)

#maximum tree depth - 1 to 6 - user input
max_depth = int(input("Enter the maximum tree depth : "))

#Class to initialize a node
class Node:
    def __init__(self, feature, depth, isleaf, label):
        self.feature = feature
        self.children = {}
        self.depth = depth
        self.isleaf = isleaf
        self.label = label

class ID3:
    def __init__(self, maxdepth,infogain):
        self.root = None
        self.maxdepth = maxdepth
        self.infogainmethod = infogain

    #function to calculate entropy of attribute values for a particular label
    @staticmethod
    def entropy(labels):
        label, counts = np.unique(labels, return_counts=True)
        prob_of_labels = counts / len(labels)
        log_of_prob = np.log2(prob_of_labels)
        ent_of_attr = 0
        for i in range(len(prob_of_labels)):
            ent_of_attr -= prob_of_labels[i] * log_of_prob[i]
        return ent_of_attr

    #function to calculate xpected entropy of an attribute
    def exp_entropy(self, data, attributes):
        exp_ent = 0
        func = None
        for value in attribute_values[attributes]:
            value_label = data[data[attributes] == value]['label']
            exp_ent += (len(value_label) / len(data)) * (ID3.entropy(value_label))
        return exp_ent

    #Function to calculate gini index of an attribute
    @staticmethod
    def giniindex(labels):
        label, counts = np.unique(labels, return_counts=True)
        prob_of_labels = counts / len(labels)
        prob_sq = [prob*prob for prob in prob_of_labels]
        return 1 - sum(prob_sq)

    # Function to calculate majority error of an attribute
    @staticmethod
    def majorityerror(labels):
        label,counts = np.unique(labels, return_counts=True)
        if len(label) == 1:
            return 0
        prob_of_labels = counts / len(labels)
        me = min(prob_of_labels)
        return me

    # Function to calculate information gain of an attribute
    def informationgain(self,data,attributes):
        infogain = 0
        for value in attribute_values[attributes]:
            value_label = data[data[attributes] == value]['label']
            if self.infogainmethod == 0:
                infogain += (len(value_label) / len(data)) * (ID3.entropy(value_label))
            elif self.infogainmethod == 1:
                infogain += (len(value_label) / len(data)) * (ID3.giniindex(value_label))
            else:
                if len(value_label) == 0:
                    return 0
                infogain += (len(value_label) / len(data)) * (ID3.majorityerror(value_label))
        return infogain

    #function to fit the decision tree
    def fit(self,data, attributes, depth):

        labels,counts = np.unique(data['label'],return_counts=True)
        #case - when all examples are of one label
        if len(labels) == 1:
            node = Node(None, depth, True, labels[0])
            return node

        #case - when no more attributes are left to split on
        if len(attributes) == 0:
            node = Node(None,depth,True,None)
            if len(labels) != 0:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel
            return node

        #case - when tree has reached its max depth
        if depth == self.maxdepth:
            node = Node(None, depth, True, None)
            if len(labels) != 0:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        node.label = leaflabel
            return node

        #case - when no more data left
        if len(data) == 0:
            return Node(None,depth,True,None)

        min_entropy = 999
        min_entropy_attr = None
        for a in attributes:
            entropy_attr = self.informationgain(data,a)
            if min_entropy > entropy_attr:
                min_entropy = entropy_attr
                min_entropy_attr = a

        #case - a non-leaf node
        node = Node(min_entropy_attr, depth, False, None)
        if self.root is None: #if it doesnt have a root node
            self.root = node
        attributes.remove(min_entropy_attr)
        for value in attribute_values[min_entropy_attr]:
            new_data = data[data[min_entropy_attr] == value]
            attr = copy.deepcopy(attributes)
            child = self.fit(new_data,attr,depth+1)
            if child.isleaf is True and child.label is None:
                maxcount = -1
                for i in range(len(counts)):
                    if counts[i] > maxcount:
                        maxcount = counts[i]
                        leaflabel = labels[i]
                        child.label = leaflabel
            node.children[value] = child
        return node

    @staticmethod
    def error(predictedlabels, targetlabels, label_values):
        y_actual = pd.Series(targetlabels,name = 'Actual')
        y_predicted = pd.Series(predictedlabels,name = 'Predicted')
        confusion_matrix = pd.crosstab(y_actual,y_predicted,margins=True)
        labels = np.unique(predictedlabels)
        correctly_predicted = 0
        for label in labels:
            correctly_predicted += confusion_matrix.loc[label,label]
        error = 1-correctly_predicted/len(predictedlabels)
        return error

    def predict(self,data_test):
        pred = []
        for i in range(len(data_test)):
            pred.append(self.leaf_label(data_test.iloc[i], self.root))
        return pred

    #returns label of a leaf node
    def leaf_label(self,data,node):
        if node.isleaf:
            return node.label
        feature_value = data[node.feature]
        child = node.children[feature_value]
        return self.leaf_label(data,child)

print("Train Errors:\n\t\t\t\tEntropy\t\tGini Index\tMajority Error\tAverage Error")
for i in range(1,max_depth+1):
    ent_dt = ID3(i, 0)
    ent_dt.fit(train_data, copy.deepcopy(attributes), 0)
    ent_pred = ent_dt.predict(train_data)
    ent_err = ID3.error(ent_pred, train_data['label'], label_values)

    gi_dt = ID3(i, 1)
    gi_dt.fit(train_data, copy.deepcopy(attributes), 0)
    gi_pred = gi_dt.predict(train_data)
    gi_err = ID3.error(gi_pred, train_data['label'], label_values)

    me_dt = ID3(i, 2)
    me_dt.fit(train_data, copy.deepcopy(attributes), 0)
    me_pred = me_dt.predict(train_data)
    maj_err = ID3.error(me_pred, train_data['label'], label_values)

    avg_err = (ent_err + gi_err + maj_err)/3
    print("For depth ", i, ":\t{:.4f}".format(ent_err), "\t\t{:.4f}".format(gi_err), "\t\t{:.4f}".format(maj_err),"\t\t\t{:.4f}".format(avg_err))

print("Test Errors: \n\t\t\t\tEntropy\t\tGini Index\tMajority Error\tAverage Error")
for i in range(1, max_depth + 1):
    ent_test_dt = ID3(i, 0)
    ent_test_dt.fit(train_data, copy.deepcopy(attributes), 0)
    ent_test_pred = ent_test_dt.predict(test_data)
    ent_err = ID3.error(ent_test_pred, test_data['label'], label_values)

    gi_test_dt = ID3(i, 1)
    gi_test_dt.fit(train_data, copy.deepcopy(attributes), 0)
    ent_test_pred = gi_test_dt.predict(test_data)
    gi_err = ID3.error(ent_test_pred, test_data['label'], label_values)

    me_test_dt = ID3(i, 2)
    me_test_dt.fit(train_data, copy.deepcopy(attributes), 0)
    me_test_pred = me_test_dt.predict(test_data)
    maj_err = ID3.error(me_test_pred, test_data['label'], label_values)

    avg_err = (ent_err + gi_err + maj_err) / 3
    print("For depth ", i, ":\t{:.4f}".format(ent_err), "\t\t{:.4f}".format(gi_err), "\t\t{:.4f}".format(maj_err),"\t\t\t{:.4f}".format(avg_err))
