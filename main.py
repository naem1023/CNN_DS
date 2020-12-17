import read_train_file
import model

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import time

from multiprocessing import Process, Lock, Queue, Pool
import multiprocessing

from tqdm import tqdm
from tqdm import trange

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from tensorflow.keras import initializers

from sklearn.model_selection import StratifiedShuffleSplit

import platform


def plot_loss_curve(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def train_model(X_train, X_test, y_train, y_test, model):
    X_train = X_train.reshape(X_train.shape[0], 300, 300, 3)
    X_test = X_test.reshape(X_test.shape[0], 300, 300, 3)

    print("X_train.shape=", X_train.shape)
    print("y_train.shape", y_train.shape)

    print("X_test.shape=", X_test.shape)
    print("y_test.shape", y_test.shape)

    # print(y_train[0])
    '''
    softmax layer -> output=10개의 노드. 각각이 0부터 9까지 숫자를 대표하는 클래스 

    이를 위해서 y값을 one-hot encoding 표현법으로 변환
    0: 1,0,0,0,0,0,0,0,0,0
    1: 0,1,0,0,0,0,0,0,0,0
    ...
    5: 0,0,0,0,0,1,0,0,0,0
    '''
    # reformat via one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # print(y_train[0])



    # catergorical_crossentropy = using when multi classficiation
    # metrics = output data type
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # batch_size : see  batch_size data and set delta in gradient decsending
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=30, verbose=1)

    plot_loss_curve(history.history)

    # print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])

    # save model in file
    # offering in KERAS
    model.save('model-201611263.model')

    history_df = pd.DataFrame(history.history)
    with open("history_data.csv", mode='w') as file:
        history_df.to_csv(file)

    return model


def get_class_name(n):
    if n == 0:
        return "food"
    elif n == 1:
        return "interior"
    elif n == 2:
        return "exterior"

def predict_image_sample(model, X_test, y_test, n):
    from random import randrange

    correct_count = 0;
    wrong_count = 0

    for idx in range(n):
        if correct_count == 2 and wrong_count == 2:
            break

        test_sample_id = randrange(len(X_test))
        test_image = X_test[test_sample_id]

        test_image = test_image.reshape(1, 300, 300, 3)

        # get answer
        y_actual = y_test[test_sample_id]

        # get prediction list
        y_pred = model.predict(test_image)

        # get prediction
        y_pred = np.argmax(y_pred, axis=1)

        # true, prediction is right
        if y_pred == y_actual and correct_count <= 2:
            plt.imshow(test_image[0])
            plt.show()

            print("==right prediction==")
            print("y_actual number=", y_actual)
            print("y_actual class=", get_class_name(y_actual))

            # 3 dimensiong
            print("y_pred number=", y_pred)
            print("y_pred number=", get_class_name(y_pred))
            print()

            correct_count += 1
        elif y_pred != y_actual and wrong_count <= 2:
            plt.imshow(test_image[0])
            plt.show()

            print("==wrong prediction==")
            print("y_actual number=", y_actual)
            print("y_actual class=", get_class_name(y_actual))

            # 3 dimensiong
            print("y_pred number=", y_pred)
            print("y_pred number=", get_class_name(y_pred))

            print()

            wrong_count += 1

    '''
    if y_pred != y_actual:
        print("sample %d is wrong!" %test_sample_id)
        with open("wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("sample %d is correct!" %test_sample_id)
    '''

def shuffle_and_valdiate(X, y):
    print("start split and shuffle!")

    shuffle_split = StratifiedShuffleSplit(train_size=0.7, test_size=0.3, n_splits=1, random_state=0)

    for train_idx, test_idx in tqdm(shuffle_split.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True, random_state=42)

    print(X_train.shape)
    print(X_test.shape)

    return X_train, X_test, y_train, y_test

def get_image():
    image_dir = 'images'

    file_number = len(os.listdir(os.path.join(image_dir)))
    print(file_number)

    # np.zeros((300, 300, 3))
    # X = np.zeros((file_number, 300, 300, 3), dtype=int)
    # #
    # y = np.zeros((file_number), dtype=int)
    X = list()
    y = list()


    for image_name in tqdm(os.listdir(os.path.join(image_dir))):
        image = cv2.imread(os.path.join(image_dir, image_name))

        if image_name[:4] == "food":
            y.append(0)

            # y[idx] = 0
        elif image_name[:8] == 'interior' :
            y.append(1)

            # y[idx] = 1
        elif image_name[:8] == 'exterior':
            y.append(2)

            # y[idx] = 2

        X.append(image)
        # X[idx] = image

    start_time = time.time()
    print("read complete")
    X = np.array(X)
    y = np.array(y)
    end_time = time.time()
    print("convert image to numpy time = ", end_time - start_time)

    print("converting complete")
    print(X.shape)
    print(y.shape)

    start_time = time.time()
    X_train, X_test, y_train, y_test = shuffle_and_valdiate(X, y)
    end_time = time.time()
    print("shuffle image time = ", end_time - start_time)

    # read_train_file.write_data(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

def make_common_model():
    model = Sequential([
        Input(shape=(300, 300, 3), name='input_layer'),

        # size of parameter = n_filters * (filter_size + 1) = 32*(9+1) = 320
        # using 32 filter
        # filter size is 3
        Conv2D(64, kernel_size=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(24, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax', name='output_layer')
    ])

    model.summary()

    return model


def make_resnet_model():

    model = Sequential()

    model.add(Input(shape=(300, 300, 3), name='input_layer'),)
    model.add(ZeroPadding2D(padding=(3,3)))

    model.add(Conv2D(32, (10, 10), strides=2, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D((2, 2), strides=1, padding='same'))

    model.add(Conv2D(32, (1, 1), strides=1, padding='valid',
                      kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model.add(MaxPooling2D((2, 2), strides=1, padding='same'))

    model.add(Conv2D(32, (1, 1), strides=2, padding='valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), strides=1, padding='same',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), strides=1, padding='valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model.add(MaxPooling2D((2, 2), strides=1, padding='same'))
    # model.add(Conv2D(8, (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='he_normal'))

    # model.add(Flatten())
    # model.add(Dense(8, activation='relu'))

    # model.add(Dropout(0.5))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(3, activation='softmax', name='output_layer'))


    model.summary()

    return model

if __name__ == '__main__':
    print(platform.architecture()[0])

    # import mnist
    #
    # mnist.train_mnist()

    all_start_time = time.time()
    start_time = time.time()
    # set tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # model = make_resnet_model()
    model = model.model_resnet()
    # model = make_common_model()

    # get train and test data
    X_train, X_test, y_train, y_test = get_image()
    print("Get all image")
    end_time = time.time()
    print("read image time = ", end_time - start_time)

    # X_train, X_test, y_train, y_test = read_train_file.read_data()
    # print("Read all image")

    # start_time = time.time()
    model = train_model(X_train, X_test, y_train, y_test, model)

    model = load_model('model-201611263.model')

    predict_image_sample(model, X_test, y_test, 500)

    end_time = time.time()
    all_end_time = time.time()

    print("train elapsed time = ", end_time - start_time)
    print("all elapsed time = ", all_end_time - all_start_time)
