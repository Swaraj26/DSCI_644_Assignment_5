import pytest
import os.path
from os import path
from titanic_a5 import file_input
from titanic_a5 import fail_file_input

def test_file_input():
    data,test_data = file_input()
    assert len(data)!=0
    assert len(test_data)!=0
    

def test_fail_file_input():
    data,test_data = fail_file_input()
    assert data!=data.empty
    assert test_data!=test_data.empty

def test_train_test_splitter():
    assert 0

def test_model_gen():
    assert 0

def test_preds():
    assert 0