import pytest
import os.path
from os import path
import pandas as pd
import titanic_a5
from titanic_a5 import file_input, model_gen, train_test_splitter
from titanic_a5 import fail_file_input
from sklearn.ensemble import RandomForestClassifier

data = file_input()
xtrain,xtest,ytrain,ytest = train_test_splitter(data)

def test_file_input():
    data= file_input()
    assert len(data)!=0
    
# def test_fail_file_input():
#     data,test_data = fail_file_input()
#     assert data!=data.empty
#     assert test_data!=test_data.empty

def test_train_test_splitter():
    shape_xtrain = (668, 5)
    shape_xtest = (223, 5)
    shape_ytrain = (668, )
    shape_ytest = (223, )
    assert shape_xtrain == xtrain.shape
    assert shape_ytrain == ytrain.shape
    assert shape_xtest == xtest.shape
    assert shape_ytest == ytest.shape



def test_preds():
    model = model_gen(xtrain,ytrain)
    testing_not_survived = pd.DataFrame({
        "Pclass":[3],
        "SibSp":[1],
        "Parch":[0],
        "Sex_female":[0],
        "Sex_male":[1]
    })
    testing_survived = pd.DataFrame({
        "Pclass":[2],
        "SibSp":[1],
        "Parch":[0],
        "Sex_female":[1],
        "Sex_male":[0]
    })
    test_prediction_1 = model.predict(testing_not_survived)
    test_prediction_2 = model.predict(testing_survived)
    assert test_prediction_1 == 0
    assert test_prediction_2 == 1