import time
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
import random
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys
import matplotlib
from sklearn import metrics


#####-----------------------------------------UnderStanding and import Data-------------------------------

#Set random seeds for reproducibility
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)


print(os.listdir("../input"))
df=pd.read_csv("../input/train.csv")
df.head()
print("Number of samples: ",len(df))
print("Number of Labels: ",np.unique(df.has_cactus))

# # plotting histogram for carat using distplot()
# sns.distplot(a=df.has_cactus)
# # # visualizing plot using matplotlib.pyplot library
# plt.show()


train=pd.read_csv("../input/train.csv")
train_images=[]
path="../input/train/"
for i in train.id:
    image=plt.imread(path+i)
    train_images.append(image)

train_images=np.asarray(train_images)
X=train_images
y=train.has_cactus
print("Labels: ",y.shape)
print("images: ",X.shape)


#####-----------------------------------------Implementation of denseNet-------------------------------

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x

#Original Function of full densenet
def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


#our Version To semi-DenseNet block
def dense_block_P_previous_layers(block_x, filters, growth_rate, layers_in_block,p):
    previous_layers = []
    previous_layers.append(block_x)
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        previous_layers.append(each_layer)
        selected_layers = previous_layers[max(0,i-p+2):i+2]
        block_x = concatenate(selected_layers, axis=-1)
        filters += growth_rate
    return block_x, filters

def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(filters, growth_rate, classes, dense_block_size, layers_in_block,p):
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block_P_previous_layers(dense_x, filters, growth_rate, layers_in_block,p)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block_P_previous_layers(dense_x, filters, growth_rate, layers_in_block,p)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(classes, activation='softmax')(dense_x)

    return Model(input_img, output)


#####-----------------------------------------Setting Data set-------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
Cat_test_y = keras.utils.to_categorical(y_test)
y_train=keras.utils.to_categorical(y_train)

print("X_train shape : ",X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",X_test.shape)
print("y_test shape : ",y_test.shape)

#####-----------------------------------------Prediction with semi-DenseNet-------------------------------

accuracy_values = {}
acc_By_diff_P_Array=[]
P_Array=[]
layers_in_block = 9
runtimes = []
total_params_array = []
for Num_of_Previous in range(1, layers_in_block+2):
    start_time = time.time()
    p = Num_of_Previous

    dense_block_size = 3
    growth_rate = 12
    classes = 2

    K.clear_session()
    model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block,p)
    summary_str=model.summary()
    total_params_value = model.count_params()
    total_params_array.append(total_params_value)
    print("total_params_array")
    print(total_params_array)

    # training
    batch_size = 32
    epochs = 8
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    history=model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, shuffle=True,validation_data=(X_test, Cat_test_y))

    end_time = time.time()
    runtime = end_time - start_time
    runtimes.append(runtime)
    print("runtimes:")
    print(runtimes)


    # Save accuracy values for each epoch in the dictionary
    accuracy_values[p] = history.history['val_accuracy']

    # set the matplotlib backend so figures can be saved in the background
    # plot the training loss and accuracy

    label_pred = model.predict(X_test)

    pred = []
    for i in range(len(label_pred)):
        pred.append(np.argmax(label_pred[i]))

    Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

    print(metrics.classification_report(Y_test, pred))



    label_pred = model.predict(X_test)

    pred = []
    for i in range(len(label_pred)):
        pred.append(np.argmax(label_pred[i]))

    Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index


    accuracy = metrics.accuracy_score(Y_test, pred)
    print(f"The accuracy in p = {Num_of_Previous}  is: {accuracy}")
    print()
    acc_By_diff_P_Array.append(accuracy)
    print("accuracy Array:")
    print(acc_By_diff_P_Array)
    print()
    P_Array.append(Num_of_Previous)
    print("Array of P already done:")
    print(P_Array)

