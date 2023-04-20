import numpy as np
import tensorflow as tf

def generate_generator_FastCoding(generator, inputdf, IMAGE_SIZE, CTU_FLAG, CTU_NN_FLAG, MOTION_FLAG,CTU_NN_Rate_Prev,rate):
    batch_size = 32
    genX1 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='CurrSL',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=batch_size,
        seed=42,
    )
    genX2 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N1PU',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX3 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N2PU',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )

    genX4 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N3PU',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX5 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N4PU',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX6 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N1SL',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX7 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N2SL',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX8 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N3SL',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX9 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='N4SL',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )

    genX10 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='rateValue',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )
    genX11 = generator.flow_from_dataframe(
        inputdf,
        "../",
        x_col='FileName',
        y_col='lfC',
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        class_mode='raw',
        batch_size=batch_size,
        seed=42,
    )

    while True:
        X1i, y1 = genX1.next()
        X2i, y2 = genX2.next()
        X3i, y3 = genX3.next()
        X4i, y4 = genX4.next()
        X6i, y6 = genX6.next()
        X7i, y7 = genX7.next()
        X8i, y8 = genX8.next()
        X10i, y10 = genX10.next()


        X5i, y5 = genX5.next()
        X9i, y9 = genX9.next()

        if (MOTION_FLAG):
            X13i, y13 = genX13.next()
            X14i, y14 = genX14.next()
            X15i, y15 = genX15.next()
            X16i, y16 = genX16.next()
            X17i, y17 = genX17.next()
            X18i, y18 = genX18.next()
            X19i, y19 = genX19.next()
            X20i, y20 = genX20.next()

            y13 = np.array(y13).astype(float)  # Need in colab due to old version of numpy
            y14 = np.array(y14).astype(float)  # Need in colab due to old version of numpy
            y15 = np.array(y15).astype(float)  # Need in colab due to old version of numpy
            y16 = np.array(y16).astype(float)  # Need in colab due to old version of numpy
            y17 = np.array(y17).astype(float)  # Need in colab due to old version of numpy
            y18 = np.array(y18).astype(float)  # Need in colab due to old version of numpy
            y19 = np.array(y19).astype(float)  # Need in colab due to old version of numpy
            y20 = np.array(y20).astype(float)  # Need in colab due to old version of numpy

            y13np = np.array(y13)
            y14np = np.array(y14)
            y15np = np.array(y15)
            y16np = np.array(y16)
            y17np = np.array(y17)
            y18np = np.array(y18)
            y19np = np.array(y19)
            y20np = np.array(y20)

            y13nx = y13np[..., np.newaxis]
            y14nx = y14np[..., np.newaxis]
            y15nx = y15np[..., np.newaxis]
            y16nx = y16np[..., np.newaxis]

            y17nx = y17np[..., np.newaxis]
            y18nx = y18np[..., np.newaxis]
            y19nx = y19np[..., np.newaxis]
            y20nx = y20np[..., np.newaxis]

        # X10i, y10 = genX8.next()
        # X11i, y11 = genX9.next()

        # X12i, y12 = genX12.next()

        y1 = np.array(y1).astype(int)  # Need in colab due to old version of numpy

        y2 = np.array(y2).astype(float)  # Need in colab due to old version of numpy
        y3 = np.array(y3).astype(float)  # Need in colab due to old version of numpy
        y4 = np.array(y4).astype(float)  # Need in colab due to old version of nump
        y5 = np.array(y5).astype(float)  # Need in colab due to old version of numpy

        y6 = np.array(y6).astype(int)  # Need in colab due to old version of nump
        y7 = np.array(y7).astype(int)  # Need in colab due to old version of numpy
        y8 = np.array(y8).astype(int)  # Need in colab due to old version of nump
        y9 = np.array(y9).astype(int)  # Need in colab due to old version of numpy

        y10 = np.array(y10).astype(float)  # Need in colab due to old version of nump

        # y11 = np.array(y11).astype(int)  # Need in colab due to old version of numpy
        # y12 = np.array(y12).astype(float)  # Need in colab due to old version of numpy

        y1np = np.array(y1)
        y2np = np.array(y2)
        y3np = np.array(y3)
        y4np = np.array(y4)
        y5np = np.array(y5)

        y6np = np.array(y6)
        y7np = np.array(y7)
        y8np = np.array(y8)
        y9np = np.array(y9)
        y10np = np.array(y10)


        y2nx = y2np[..., np.newaxis] # PU of Neighbour 1
        y3nx = y3np[..., np.newaxis] # PU of Neighbour 2
        y4nx = y4np[..., np.newaxis] # PU of Neighbour 3
        y5nx = y5np[..., np.newaxis] # PU of Neighbour 4

        y6nx = y6np[..., np.newaxis] # Split info of Neighbour 1
        y7nx = y7np[..., np.newaxis] # Split info of Neighbour 2
        y8nx = y8np[..., np.newaxis] # Split info of Neighbour 3
        y9nx = y9np[..., np.newaxis] # Split info of Neighbour 4

        y10nx = y10np[..., np.newaxis] # This is Rate value (Qp Value normalized between 0 - 1 )


        VecData = np.array(np.concatenate((y2nx, y3nx, y4nx,y5nx, y6nx, y7nx, y8nx,y9nx,y10nx), axis=1))


        if (CTU_NN_FLAG):
            yield [X1i, VecData], y1  # CTU + CNN + Dense
        elif CTU_NN_Rate_Prev:
            yield [X1i, y10nx], y1
        elif (CTU_FLAG):
            # Tried to append rate as a tensor in CNN for better learning, not good results
            # qpmatrix = np.full((batch_size, 64, 64, 1), (int(rate) / 50))
            # darray = np.concatenate((X1i, qpmatrix), axis=3)
            # yield darray, y1
            # Another Try where concatenating rate + previous layer tensors
            # rateValue = (int(rate) - 8) / (45 - 8)
            # VecData = np.array(y10nx)


            yield X1i, y1  # CNN only
        else:
            yield VecData, y1  # Feature vector only


class MultiChannelDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size,
                 shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)
        self.n_type = df['CurrSL'].nunique()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path0, path1, path2, path3, path4, target_size):
        image0 = tf.keras.preprocessing.image.load_img(path0, color_mode='grayscale')
        image_arr0 = tf.keras.preprocessing.image.img_to_array(image0)
        image_arr0 = tf.image.resize(image_arr0, (target_size[0], target_size[1])).numpy()
        image_arr0 = image_arr0 / 255.

        image1 = tf.keras.preprocessing.image.load_img(path1, color_mode='grayscale')
        image_arr1 = tf.keras.preprocessing.image.img_to_array(image1)
        image_arr1 = tf.image.resize(image_arr1, (target_size[0], target_size[1])).numpy()
        image_arr1 = image_arr1 / 255.

        image2 = tf.keras.preprocessing.image.load_img(path2, color_mode='grayscale')
        image_arr2 = tf.keras.preprocessing.image.img_to_array(image2)
        image_arr2 = tf.image.resize(image_arr2, (target_size[0], target_size[1])).numpy()
        image_arr2 = image_arr2 / 255.

        image3 = tf.keras.preprocessing.image.load_img(path3, color_mode='grayscale')
        image_arr3 = tf.keras.preprocessing.image.img_to_array(image3)
        image_arr3 = tf.image.resize(image_arr3, (target_size[0], target_size[1])).numpy()
        image_arr3 = image_arr3 / 255.

        image4 = tf.keras.preprocessing.image.load_img(path4, color_mode='grayscale')
        image_arr4 = tf.keras.preprocessing.image.img_to_array(image4)
        image_arr4 = tf.image.resize(image_arr4, (target_size[0], target_size[1])).numpy()
        image_arr4 = image_arr4 / 255.

        img_arr = np.concatenate((image_arr0, image_arr1, image_arr2, image_arr3, image_arr4), axis=-1)
        return img_arr

    def __get_output(self, label, num_classes):
        # return tf.keras.utils.to_categorical(label, num_classes=num_classes)
        return np.array(int(label))

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path0_batch = batches[self.X_col['Cur_CTU']]
        path1_batch = batches[self.X_col['N1_CTU']]
        path2_batch = batches[self.X_col['N2_CTU']]
        path3_batch = batches[self.X_col['N3_CTU']]
        path4_batch = batches[self.X_col['N4_CTU']]
        pu0_batch = np.array(batches[self.X_col['N1PU']]).astype(float)
        pu1_batch = np.array(batches[self.X_col['N2PU']]).astype(float)
        pu2_batch = np.array(batches[self.X_col['N3PU']]).astype(float)
        pu3_batch = np.array(batches[self.X_col['N4PU']]).astype(float)
        sl0_batch = np.array(batches[self.X_col['N1SL']]).astype(float)
        sl1_batch = np.array(batches[self.X_col['N2SL']]).astype(float)
        sl2_batch = np.array(batches[self.X_col['N3SL']]).astype(float)
        sl3_batch = np.array(batches[self.X_col['N4SL']]).astype(float)

        type_batch = batches[self.y_col['type']]

        # vectorTuple = np.asarray(tf.transpose(
        #     tuple([pu0_batch, pu1_batch, pu2_batch, pu3_batch, sl0_batch, sl1_batch, sl2_batch, sl3_batch])))

        X_batch = np.asarray([self.__get_input(x0, x1, x2, x3, x4, self.input_size) for x0, x1, x2, x3, x4 in
                              zip(path0_batch, path1_batch, path2_batch, path3_batch, path4_batch)])
        y_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, y_batch
        #return tuple([X_batch, vectorTuple]), y_batch
        # return X_batch, tuple([y0_batch, y1_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


