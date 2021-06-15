import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

np_array = np.array([[28,'management','single','university.degree','no','yes','no','cellular','jun','thu',339,3,6,2,'success',-1.7,94.055,-39.8,0.729,4991.6]])
test = pd.DataFrame(np_array, columns=['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed'])



df = pd.read_csv("banking.csv")

X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week' , 'duration', 'campaign', 'pdays', 'previous' , 'poutcome' , 'emp_var_rate' , 'cons_price_idx' , 'cons_conf_idx' , 'euribor3m' , 'nr_employed']]
y=df[['y']]

print("Main Dataframe: ")
print(y.value_counts())

print('******************************')

print(df['education'].unique())
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])

print('******************************')

count_no_sub = len(df[df['y']==0])
count_sub = len(df[df['y']==1])

percent_of_no_sub = count_no_sub/(count_no_sub+count_sub)
percent_of_sub = count_sub/(count_no_sub+count_sub)

print("No subscription: ",count_no_sub)
print("Subscription: ",count_sub)

print("percentage of no subscription is", percent_of_no_sub*100)
print("percentage of subscription", percent_of_sub*100)

print("************************************")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print("Train data count: ")
print(y_train.value_counts())

print("Test Data Count: ")
print(y_test.value_counts())

print("****************")


# label encode the categorical values and convert them to numbers
for column in X_train:
    if X_train[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(X_train[column].astype(str))
        X_train[column] = le.transform(X_train[column].astype(str))
        X_test[column] = le.transform(X_test[column].astype(str))
        test[column]=le.transform(test[column].astype(str))


# Smote
smt = SMOTE('minority',random_state=0)
X_train,y_train = smt.fit_resample(X_train, y_train)

#Logistic Regression
logRegression = LogisticRegression()
logRegression.fit(X_train,y_train)
y_predictionLogistic = logRegression.predict(X_test)


print("Testing")

print(test)

testingPrediction=logRegression.predict(test)

print(testingPrediction)