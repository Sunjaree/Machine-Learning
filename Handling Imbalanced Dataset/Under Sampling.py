import numpy as np
import pandas as pd


from sklearn import metrics
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("data.csv")


#Under Sampling
df_class_0 = df[df['Y']==0]
df_class_1 = df[df['Y']==1]
df_class_0_count,df_class_1_count = df.Y.value_counts()
df_class_0_under = df_class_0.sample(df_class_1_count)
df = pd.concat([df_class_1,df_class_0_under],axis = 0)



df = df.drop(['Serial'],axis = 1)
X = df.drop('Y',axis= 1)
y=df[['Y']]


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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)

#Logistic Regression
print("Logistic Regression: ")
logRegression = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=60)
logRegression.fit(X_train,y_train)
y_predictionLogistic = logRegression.predict(X_test)
print(metrics.confusion_matrix(y_test,y_predictionLogistic))
print("Accuracy: ",metrics.accuracy_score(y_test, y_predictionLogistic)*100)

print(classification_report(y_test,y_predictionLogistic))