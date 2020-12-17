import os
import numpy as np
from tqdm import tqdm

def write_data(X_train, X_test, y_train, y_test):
    X_train_file = open(os.path.join("train_data", "X_train.txt"), 'w')
    X_test_file = open(os.path.join("train_data", "X_test.txt"), 'w')
    y_train_file = open(os.path.join("train_data", "y_train.txt"), 'w')
    y_test_file = open(os.path.join("train_data", "y_test.txt"), 'w')

    for i in tqdm(range(X_train.shape[0])):
        y_train_item = str(y_train[i]) + '\n'

        y_train_file.write(y_train_item)

        for j in range(X_train.shape[1]):
            for k in range(X_train.shape[2]):
                X_train_item = str(X_train[i][j][k][0]) + "," + str(X_train[i][j][k][1]) + "," + str(X_train[i][j][k][2]) + '\n'

                X_train_file.write(X_train_item)

    for i in tqdm(range(X_test.shape[0])):
        y_test_item = str(y_test[i]) + '\n'

        y_test_file.write(y_test_item)

        for j in range(X_test.shape[1]):
            for k in range(X_test.shape[2]):
                X_test_item = str(X_test[i][j][k][0]) + "," + str(X_test[i][j][k][1]) + "," + str(X_test[i][j][k][2]) + '\n'

                X_test_file.write(X_test_item)


    X_train_file.close()
    X_test_file.close()
    y_train_file.close()
    y_test_file.close()

def read_data():
    X_train_file = open(os.path.join("train_data", "X_train.txt"), 'r')
    X_test_file = open(os.path.join("train_data", "X_test.txt"), 'r')
    y_train_file = open(os.path.join("train_data", "y_train.txt"), 'r')
    y_test_file = open(os.path.join("train_data", "y_test.txt"), 'r')

    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()

    # (n, 300 ,300, 3)
    #  X  temp1 temp2
    for image_X_train, image_X_test, image_y_train, image_y_test \
        in tqdm(zip(X_train_file.readlines(), X_test_file.readlines(), y_train_file.readlines(), y_test_file.readlines())):

        X_train_temp1 = list()
        X_test_temp1 = list()
        y_train_temp1 = list()
        y_test_temp1 = list()

        for i in range(300):
            X_train_temp2 = list()
            X_test_temp2 = list()

            for j in range(300):
                for val in image_X_train.split(','):
                    X_train_temp2.append(val)

                for val in image_X_test.split(','):
                    X_test_temp2.append(val)

            X_train_temp1.append(X_train_temp2)
            X_test_temp1.append(X_test_temp2)

        y_train_temp1.append(val[0])
        y_test_temp1.append(val[0])

        X_train.append(X_train_temp1)
        X_test.append(X_test_temp1)
        y_train.append(y_train_temp1)
        y_test.append(y_test_temp1)

    X_train_file.close()
    X_test_file.close()
    y_train_file.close()
    y_test_file.close()

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test