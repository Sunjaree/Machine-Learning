import numpy as np
import pandas as pd


from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("data.csv")
df = df.drop(['Serial'],axis = 1)



X = df.drop('Y',axis= 1)
y=df[['Y']]

#print(abs(df.corr()["Y"]))

print("Main Dataframe: ")
print(y.value_counts())

print('******************************')

count_no_default = len(df[df['Y']==0])
count_default = len(df[df['Y']==1])

percent_of_no_default = count_no_default / (count_no_default + count_default)
percent_of_default = count_default / (count_no_default + count_default)

print("No Default: ",count_no_default)
print("Default: ",count_default)

print("percentage of no Default is: ", percent_of_no_default*100)
print("percentage of Default is: ", percent_of_default*100)

print("************************************")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#normalization
from  sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# # label encode the categorical values and convert them to numbers
# for column in X_train:
#     if X_train[column].dtype == 'object':
#         le = LabelEncoder()
#         le.fit(X_train[column].astype(str))
#         X_train[column] = le.transform(X_train[column].astype(str))
#         X_test[column] = le.transform(X_test[column].astype(str))


# Smote
smt = SMOTE('minority',random_state=0)
X_train,y_train = smt.fit_resample(X_train, y_train)


#Logistic Regression
print("Logistic Regression: ")
logRegression = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=60,class_weight={0:34.3,1:100})
logRegression.fit(X_train,y_train)
y_predictionLogistic = logRegression.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionLogistic))
print("Accuracy: ",metrics.accuracy_score(y_test, y_predictionLogistic)*100)

print("******************************")

from sklearn.ensemble import RandomForestClassifier
print("Random Forest Classifier: ")
RandomClassifier = RandomForestClassifier()
RandomClassifier.fit(X_train, y_train)
y_predictionRandomForest = RandomClassifier.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionRandomForest))
print("accuracy: ", metrics.accuracy_score(y_test,y_predictionRandomForest) * 100)


print("******************************")

from sklearn import tree
print("Decision Tree Classifier: ")
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(X_train,y_train)
y_predictionDecisionTree = DecisionTree.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionDecisionTree))
print("accuracy: ", metrics.accuracy_score(y_test,y_predictionDecisionTree) * 100)


print("******************************")

from sklearn.neighbors import KNeighborsClassifier
print("K-Nearest Neighbour Classifier: ")
KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train,y_train)
y_predictionKNN = KNN.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionKNN))
print("accuracy: ", metrics.accuracy_score(y_test,y_predictionKNN) * 100)

print("******************************")

from sklearn.ensemble import AdaBoostClassifier
print("AdaBoost Classifier: ")
AdaBoost = AdaBoostClassifier(n_estimators=100,learning_rate=1)
AdaBoost.fit(X_train,y_train)
y_predictionAdaBoost = AdaBoost.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionAdaBoost))
print("accuracy: ", metrics.accuracy_score(y_test,y_predictionAdaBoost) * 100)

print("******************************")

from sklearn.neural_network import MLPClassifier
print("Multi Layer Perceptron Classifier: ")
MLP = MLPClassifier(max_iter=100,activation='relu')
MLP.fit(X_train,y_train)
y_predictionMLP = MLP.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionMLP))
print("accuracy: ", metrics.accuracy_score(y_test,y_predictionMLP) * 100)


print("******************************")

from  sklearn.naive_bayes import GaussianNB
print("Gaussian Naive Bayes classifier: ")
Gaussian = GaussianNB()
Gaussian.fit(X_train, y_train)
y_predictionGaussian = Gaussian.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predictionGaussian))
print("accuracy: ", metrics.accuracy_score(y_test, y_predictionGaussian) * 100)



