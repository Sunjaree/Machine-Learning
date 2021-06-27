import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 35)


df = pd.read_csv("Cancer.csv")
X = df.drop(['id','diagnosis','Unnamed: 32'], axis = 1)
Y = df['diagnosis']

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

#hyperparameter Optomization using Grid SearchCV
from sklearn.model_selection import GridSearchCV
logRegression = LogisticRegression()
param_grid = [

    {'penalty' : ['l1','l2','none'],
    #'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','saga'],
    'max_iter' : [100, 1000]
    }
]
clf = GridSearchCV(logRegression,param_grid = param_grid,cv=3,verbose=True,n_jobs=-1)
clf.fit(X,Y)
best_clf_pred = clf.predict(X_test)


print(metrics.confusion_matrix(y_test, best_clf_pred))
print("accuracy: ", metrics.accuracy_score(y_test, best_clf_pred) * 100)