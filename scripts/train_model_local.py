'''
Train and save the model.
Take in training data src path (csv) and model name as command line arguments.

ex. usage - python3 train_model.py ../data/full_train.csv local-full;
'''
import sys

import numpy as np 
import pandas as pd 
import xgboost as xgb

def load_training_data(src_path):
    dataset = pd.read_csv(src_path, delimiter=',', header=0)
    data, label = dataset.iloc[:,:-1], dataset.iloc[:,-1:]
    full_dtrain = xgb.DMatrix(data, label=label)
    return full_dtrain

def train(src_path, model_name):
    dtrain = load_training_data(src_path)
    params = {'max_depth': 3,   # default = 6
              'alpha': 0,       # default = 0
              'lambda': 1,      # default = 1
              'eta': 0.3,       # default = 0.3
              'gamma': 0.3,     # default = 0
              'objective': 'binary:hinge'
            }
    num_rounds = 25

    model = xgb.train(params, dtrain, num_rounds)
    model.save_model(''.join(['../models/', str(model_name), '.model']))
    model.dump_model(''.join(['../models/', str(model_name), '.txt_dump']))
    print('saved and dumped model for', model_name)
    
def main(argv):
    assert(len(argv) >= 3), "This script takes in source path of training data and model name."
    src_path = argv[1]
    model_name = argv[2]
    train(src_path, model_name)

if __name__ == '__main__':
    main(sys.argv)
