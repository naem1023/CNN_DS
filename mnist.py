'''
MNIST: 미국표준기술연구소에 공개한 필기체 숫자에 대한 데이터베이스
'''

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.datasets import mnist

import tensorflow as tf


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


def train_mnist_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    '''
    print("X_train.shape=", X_train.shape)
    print("y_train.shape", y_train.shape)

    print("X_test.shape=", X_test.shape)
    print("y_test.shape", y_test.shape)

    print(y_train[1])
    plt.imshow(X_train[1], cmap='gray')
    '''

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    print(y_train[0])
    '''
    softmax layer -> output=10개의 노드. 각각이 0부터 9까지 숫자를 대표하는 클래스 

    이를 위해서 y값을 one-hot encoding 표현법으로 변환
    0: 1,0,0,0,0,0,0,0,0,0
    1: 0,1,0,0,0,0,0,0,0,0
    ...
    5: 0,0,0,0,0,1,0,0,0,0
    '''

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_train[0])

    model = Sequential([
        Input(shape=(28, 28, 1), name='input_layer'),

        # size of parameter = n_filters * (filter_size + 1) = 32*(9+1) = 320
        # using 32 filter
        # filter size is 3
        Conv2D(16, kernel_size=(1,1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(8, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(1, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(8, kernel_size=(3, 3)),
        BatchNormalization(),
        Activation('relu'),

        Flatten(),

        Dense(8, activation='relu'),
        Dense(10, activation='softmax', name='output_layer')
    ])

    model.summary()

    # catergorical_crossentropy = using when multi classficiation
    # metrics = output data type
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # batch_size : see  batch_size data and set delta in gradient decsending
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=3)
    plot_loss_curve(history.history)

    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])

    # save model in file
    # offering in KERAS
    model.save('mnist.model')

    return model


def predict_image_sample(model, X_test, y_test, test_id=-1):
    # test id : set data to test
    test_sample_id = 0
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(10000)
    else:
        test_sample_id = test_id

    test_image = X_test[test_sample_id]

    plt.imshow(test_image, cmap='gray')
    plt.show()

    test_image = test_image.reshape(1, 28, 28, 1)

    y_actual = y_test[test_sample_id]
    print("y_actual number=", y_actual)

    y_pred = model.predict(test_image)

    # 10 dimension
    print("y_pred=", y_pred)

    y_pred = np.argmax(y_pred, axis=1)[0]
    print("y_pred number=", y_pred)

    '''
    if y_pred != y_actual:
        print("sample %d is wrong!" %test_sample_id)
        with open("wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("sample %d is correct!" %test_sample_id)
    '''


def train_mnist():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    model = train_mnist_model()
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # model = load_model('mnist.model')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    predict_image_sample(model, X_test, y_test)
    # for i in range(500):
    #    predict_image_sample(model, X_test, y_test)
