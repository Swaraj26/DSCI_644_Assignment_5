import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/train.csv")
test_data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/test.csv")

y = data['Survived']
features = ['Pclass','Sex','SibSp','Parch']
x = pd.get_dummies(data[features])
# X_test = pd.get_dummies(test_data[features])
# y_test = test_data[]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=420)

model = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=69)

model.fit(xtrain,ytrain)
predictions = model.predict(xtest)

print("Accuracy = ",round((accuracy_score(ytest,predictions)*100),4),"%") 
print(" ")