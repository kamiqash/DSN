from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout,Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate


def Kim_NN_CTU_FV_D0(img_input_shape,rate_input):
    input_img = Input(img_input_shape, name='img')
    c1 = Conv2D(32, (29, 29), activation='relu', padding='valid')(input_img)  # c1 = 36,36,32
    p1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='valid')(c1)  # p1 = 32, 32, 32

    c2 = Conv2D(64, (13, 13), activation='relu', padding='valid')(p1)  # c2 = 20,20,64
    p2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(c2)  # p2 = 16, 16, 64

    c3 = Conv2D(128, (5, 5), activation='relu', padding='valid')(p2)  # c3 = 12,12,128
    p3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu', padding='valid')(p3)  # c3 = 12,12,128
    p4 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='valid')(c4)
    gp = Flatten()(p4)
    z = Dense(256, activation="relu")(gp)
    z = Concatenate(axis=1)([z, rate_input])
    InpArray = [input_img, rate_input]

    return z,InpArray

def Basic_NN_CTU_FV_D0(img_input_shape,rate_input):

    input_img = Input(img_input_shape, name='img')

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(c2)  # p1 = 32, 32, 32

    c3 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv3')(p1)  # c2 = 20,20,64
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(c3)
    p2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(c4)  # p2 = 16, 16, 64

    flat1 = Flatten(name='flat1')(p2)
    ds1 = Dense(64, name='dense1')(flat1)
    z = Concatenate(axis=1)([ds1, rate_input])
    z = Dense(32, activation="relu")(z)
    z = Dense(1, activation="sigmoid")(z)

    InpArray = [input_img, rate_input]
    return z, InpArray


def Basic_NN_CTU_D0(img_input_shape):

    input_img = Input(img_input_shape, name='img')

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(c2)  # p1 = 32, 32, 32

    c3 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv3')(p1)  # c2 = 20,20,64
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(c3)
    p2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(c4)  # p2 = 16, 16, 64

    flat1 = Flatten(name='flat1')(p2)
    ds1 = Dense(64, name='dense1')(flat1)
    z = Dense(32, activation="relu")(ds1)
    z = Dense(1, activation="sigmoid")(z)

    InpArray = [input_img]
    return z, InpArray


def NN_FV(rate_input):
    InpArray = [rate_input]
    z = Dense(256, activation="relu")(rate_input)
    z = Dense(128, activation="relu")(z)
    z = Dense(32, activation="relu")(z)
    z = Dense(1, activation="sigmoid")(z)
    return z, InpArray

