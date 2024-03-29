'''
Load a model and evaluate it on test data.
Takes in model file path from command line argument.
Outputs predictions and evaluation in ../predictions/ directory.

python3 predict_and_eval.py <model file path> <test data file path>
'''
import sys 

import numpy as np
import pandas as pd
import xgboost as xgb

def get_fp_fn(predictions, test_labels):
    false_positives = 0
    false_negatives = 0
    total = 0
    for p, l in np.column_stack((predictions, test_labels)):
        if p == 0 and l == 1:
            false_negatives += 1
        if p == 1 and l == 0:
            false_positives += 1
        total += 1
    return str({
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total predictions': total
    })

def main(argv):
    assert(len(argv) == 3), "This script takes in the path of the model and path of the training data as command line arguments."
    model_path = argv[1]
    model_name = model_path[10:-6]
    test_set_path = argv[2]
    test_set_name = test_set_path[8:-4]

    dataset = pd.read_csv(test_set_path, delimiter=',', header=0)
    data, label = dataset.iloc[:,:-1], dataset.iloc[:,-1:]
    dtest = xgb.DMatrix(data, label=label)
    test_labels = dtest.get_label()

    model = xgb.Booster(model_file=model_path)
    predictions = model.predict(dtest)
    evaluation = model.eval(dtest)
    
    predict_filepath = ''.join(['../predictions/', model_name, '_', test_set_name, '.predict'])
    np.savetxt(predict_filepath, predictions)

    f = open(predict_filepath, 'a')
    f.write(evaluation)
    f.write('\n')
    f.write(get_fp_fn(predictions, test_labels))
    f.close()

    print('saved predictions and evaluations for', model_path)


if __name__ == '__main__':
    main(sys.argv)