#!/bin/bash

# local training full dataset
python3 train_model_local.py ../data/full_train.csv local-full;

# local training partitioned dataset
python3 train_model_local.py ../data/tenth_train.csv local-partitioned;

# federated training full dataset
../../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers 10 \
python3 train_model_federated.py ../data/full_train.csv federated-full;

########################################################################

# test full local against full test
python3 predict_and_eval.py ../models/local-full.model ../data/full_test.csv;

# test partitioned local against partitioned test
python3 predict_and_eval.py ../models/local-partitioned.model ../data/tenth_test.csv;

# test partitioned local against full test
python3 predict_and_eval.py ../models/local-partitioned.model ../data/full_test.csv;

# test federated against full test
python3 predict_and_eval.py ../models/federated-full.model ../data/full_test.csv;