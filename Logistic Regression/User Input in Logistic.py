import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


User = np.array([[28, 'management', 'single', 'university.degree', 'no', 'yes', 'no', 'cellular', 'jun', 'thu', 339, 3, 6, 2, 'success', -1.7, 94.055, -39.8, 0.729, 4991.6]])
User_Input = pd.DataFrame(User, columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'])

df = pd.read_csv("banking.csv")

X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week' , 'duration', 'campaign', 'pdays', 'previous' , 'poutcome' , 'emp_var_rate' , 'cons_price_idx' , 'cons_conf_idx' , 'euribor3m' , 'nr_employed']]
y=df[['y']]

print("User Input: ")
print(User_Input)

print('******************************')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print("****************")

# label encode the categorical values and convert them to numbers
for column in X_train:
    if X_train[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(X_train[column].astype(str))
        X_train[column] = le.transform(X_train[column].astype(str))
        X_test[column] = le.transform(X_test[column].astype(str))
        User_Input[column]=le.transform(User_Input[column].astype(str))


# Smote
smt = SMOTE('minority',random_state=0)
X_train,y_train = smt.fit_resample(X_train, y_train)

#Logistic Regression
logRegression = LogisticRegression()
logRegression.fit(X_train,y_train)
y_predictionLogistic = logRegression.predict(X_test)

testingPrediction = logRegression.predict(User_Input)
print("Predicted Result: ",testingPrediction[0])

if testingPrediction[0]==1:
    print("This person will subscribe :)")
else:
    print("This person will not subscribe :(")
