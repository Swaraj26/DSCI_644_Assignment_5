import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def fail_file_input():
    data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/training.csv")
    test_data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/test.csv")
    return data,test_data

def file_input():
    data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/train.csv")
    test_data = pd.read_csv("C:/Users/swara/OneDrive/HW/SWDS/hw/Assignment_5/test.csv")
    return data

def train_test_splitter(data):
    y = data['Survived']
    features = ['Pclass','Sex','SibSp','Parch']
    x = pd.get_dummies(data[features])

    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=420)
    return xtrain,xtest,ytrain,ytest

def model_gen(xtrain,ytrain):
    model = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=69)
    model.fit(xtrain,ytrain)
    return model

def preds(model,xtest,ytest):
    predictions = model.predict(xtest)
    print("Accuracy = ",round((accuracy_score(ytest,predictions)*100),4),"%") 

def main():
    data= file_input()
    xtrain,xtest,ytrain,ytest = train_test_splitter(data)
    model = model_gen(xtrain,ytrain)
    print(model)
    preds(model,xtest,ytest)

if __name__ == "__main__":
    main()

