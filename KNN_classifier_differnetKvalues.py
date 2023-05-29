# -*- coding: utf-8 -*-
"""
Implementation of the KNN algorithm for classification

"""

###########Loading iris dataset from sklearn #################
# import the datasets from sklearn
from sklearn import datasets

#lets's import the iris dataset
iris = datasets.load_iris()

# show the data in iris
print(iris.data)
#show the labels (target) of iris
print(iris.target)

#print the shape of the data in iris
print(iris.data.shape)
#print the shape of the labels(target) in iris
print(iris.target.shape)

#print put the features of the data
print(iris.feature_names)
# show the classes in the target (labels)
print(iris.target_names)

#Let see the type fo the iris data and iris target
print(type(iris.data))
print(type(iris.target))
###############################################################################

######### divide data to train and testing######################

#put the data and the targets in individual arrays
# f(X)= y = output X corresponds to data and y to target
X = iris.data # X contains the data from iris dataset only (features)
y= iris.target# y contain the corresponding targets of the data from iris (target =>labels)

#splitting the dataset into train and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 4)


#print the shape of the training and testing array
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

##############################################################################

###### implementing the KNN classifier
"""
Full KNN documentation 
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
#import the KNN classifier function
from sklearn.neighbors import KNeighborsClassifier
#import the metrics module to check the accuracy of the model
from sklearn import metrics

# try K values from 1 to 26 to find the best k value
#Create the classifier model

"""
k_values = range (1,26)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test) 
    score = metrics.accuracy_score(y_test, y_pred)
    print(" For k =", k, "the accuracy =", score)


"""



#THis code in case we want to save the result (scores in a list) for all k values
k_values = range(1,26)
scores = {}
scores_list = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors= k)
    #start training the KNN
    knn.fit(X_train, y_train)
    #test our KNN model
    y_pred = knn.predict(X_test) # y_pred is the actual classificiation decision of the KNN model
                                # based on the testing data X_test
    #measure the accuracy of our model
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
#print out the model accuracy
print(scores_list)
































