#Importing all the libraries required
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

#test function to test if the file input works and doesnt get a empty file error
def test_file_input():
    data= file_input()
    assert len(data)!=0

#the inital failing test function, just to see how it all works and how to resolve things, but is commented out to get all the tests as passed
# def test_fail_file_input():
#     data,test_data = fail_file_input()
#     assert data!=data.empty
#     assert test_data!=test_data.empty

#test function to test if the splitter function has split the dataset into the desired shape and not have any empty datasets by mistake
def test_train_test_splitter():
    shape_xtrain = (668, 5)
    shape_xtest = (223, 5)
    shape_ytrain = (668, )
    shape_ytest = (223, )
    assert shape_xtrain == xtrain.shape
    assert shape_ytrain == ytrain.shape
    assert shape_xtest == xtest.shape
    assert shape_ytest == ytest.shape


#test function to test the predictions of the model, with 2 different inputs with both of them being with known outputs, which is compared to see if the model predicts the things correctly or not.
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