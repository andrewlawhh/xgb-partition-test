import pandas as pd

def split_train():
    full_train = pd.read_csv('../data/full_train.csv', delimiter=',', header=0)
    tenth_train_rows = len(full_train) // 10
    tenth_train = full_train.iloc[:tenth_train_rows, :]
    tenth_train.to_csv('../data/tenth_train.csv', index=False)

def split_test():
    full_test = pd.read_csv('../data/full_test.csv', delimiter=',', header=0)
    tenth_test_rows = len(full_test) // 10
    tenth_test = full_test.iloc[:tenth_test_rows, :]
    tenth_test.to_csv('../data/tenth_test.csv', index=False)

def main():
    split_train()
    split_test()

if __name__ == '__main__':
    main()
