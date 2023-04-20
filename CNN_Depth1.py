
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

import os
import numpy as np
import time
#from google.colab import files
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

import tensorflow as tf

from CustomGenerator import generate_generator_FastCoding
from CustomGenerator import MultiChannelDataGen


from LoadLFImages_Backup import LoadFiftySixViewsDepth1

from ML_Schemes import Kim_NN_CTU_FV_D0, Basic_NN_CTU_rate
from ML_Schemes import Basic_NN_CTU_FV_D0
from ML_Schemes import Basic_NN_CTU_D0
from ML_Schemes import NN_FV


from UtilityFuncs import Stats_Split_Non_Split

import random
def Depth1_CNN(loopidx,CTU_FLAG,MOTION_FLAG,CTU_NN_FLAG,rate_list,DBlist,PRINT_STATUS,PRINT_STATS,TestInfo,EPOCHS,outputFolder,SYSTEM,MULTI_CTU_FLAG,METHODOLOGY,PreDefinedTest,DEBUG,CTU_NN_Rate_Prev):

    Record_Rates=[]
    # LF_list=[29, 38] # Total LF images in Dataset
    LF_list = [28]  # Total LF images in Dataset
    counter=0
    for DB in DBlist:
        TotLF = LF_list[counter]
        counter=counter+1
        for rate in rate_list:
            Record_Rates = []
            if (PRINT_STATUS):
                print("Depth 1: Processing Rate: %s, FLAGS CTU_FLAG; %d, CTU_NN_FLAG: %d , TEST: %s" % (
                rate, CTU_FLAG, CTU_NN_FLAG, TestInfo))

            IMAGE_SIZE = (32, 32)
            batch_size = 32

            pathDrive = outputFolder +'Depth1/QP_' + str(rate) + '/'

            if not os.path.exists(pathDrive):
                os.makedirs(pathDrive)
            #sys.path.append(pathDrive)


            # if  (SYSTEM==1):
            #     physical_devices = tf.config.list_physical_devices('GPU')
            #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
            #     tf.config.experimental.set_memory_growth(physical_devices[1], True)

            if (SYSTEM == 2):
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)

            if (PreDefinedTest):
                # Pre defined three Test Images selected based on distribution of split and non split ratio
                randomTestData_22 = [['4', '10', '13'], ['14', '23', '27'], ['5', '7', '16']]
                randomTestData_27 = [['5', '10', '25'], ['15', '19', '22'], ['6', '12', '26']]
                randomTestData_32 = [['9', '17', '19'], ['6', '12', '22'], ['3', '6', '12']]

                if TestInfo == 'A':
                    index=0
                elif TestInfo == 'B':
                    index=1
                elif TestInfo == 'C':
                    index=2
                if (DEBUG):
                    TotLF = 4
                    randomTestData = ['2']
                elif (rate == '22'):
                    randomTestData = randomTestData_22[index]
                elif (rate == '27'):
                    randomTestData = randomTestData_27[index]
                elif (rate == '32'):
                    randomTestData = randomTestData_32[index]
            else:
                # Defining Random 3 LF images as test Images
                randomTestData = []
                for rn in range(3):
                    ranvalue = random.randint(1, TotLF)
                    randomTestData.append(str(ranvalue))


            if SYSTEM == 3:
                metaPath = '/content/' + rate + '_META/'
                ctuPath = '/content/CTU/'

            elif SYSTEM==1:
                metaPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth0/' + rate + '_META/'
                ctuPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth0/CTU/'
                metaPath = 'D:\\CODECS\\FAST Coding\\Dataset\\' + DB + '\\Depth1\\META_' + rate + '\\'
                ctuPath = 'D:\\CODECS\\FAST Coding\\Dataset\\' + DB + '\\Depth1\\CTU_' + rate + '\\'

            elif SYSTEM==2:
                # metaPath= 'C:\\Research Work\\FastCodingProject\\Dataset\\' +DB + '\\Depth1\\META_' + rate + '\\'
                # ctuPath =  'C:\\Research Work\\FastCodingProject\\Dataset\\' +DB + '\\Depth1\\CTU_' + rate + '\\'
                # metaPath = 'D:\\CODECS\\FAST Coding\\Dataset\\' + DB + '\\Depth1\\META_' + rate + '\\'
                # ctuPath = 'D:\\CODECS\\FAST Coding\\Dataset\\' +DB + '\\Depth1\\CTU_' + rate + '\\'
                ctuPath = 'C:\\PHD\\Dataset2022\\Depth1\\CTU_' + rate + '\\'
                metaPath = 'C:\\PHD\\Dataset2022\\Depth1\\META_' + rate + '\\'
            # Read 56 views out of 81 views
            Record=LoadFiftySixViewsDepth1(TotLF, ctuPath, metaPath, rate, PRINT_STATUS,METHODOLOGY,DB)
            Record_Rates.extend(Record)
    # From Here
            rate1Df = pd.DataFrame(np.array(Record_Rates),
                              columns=['FileName', 'LFname', 'lfR', 'lfC', 'rateValue', 'CTUno', 'CurrPU', 'CurrSL', 'N1PU',
                                       'N2PU', 'N3PU', 'N4PU', 'N1SL', 'N2SL', 'N3SL', 'N4SL', 'N1_CTU', 'N2_CTU', 'N3_CTU',
                                       'N4_CTU', 'viewNumber','PartNo'])

            print("Dataset is read with Total Samples: ",len(rate1Df))

            # Splitting test and train data
            train_df = rate1Df[~rate1Df['LFname'].isin(randomTestData)]
            validate_df = rate1Df[rate1Df['LFname'].isin(randomTestData)]
            train_df = train_df.reset_index(drop=True)  # this reset the list so that it start from 0 to its end
            validate_df = validate_df.reset_index(drop=True)  # this reset the list so that it start from 0 to its e

            # Stats of Test and Train data
            if(PRINT_STATS):
                Stats_Split_Non_Split(rate1Df,     "[Input Data   :]")
                Stats_Split_Non_Split(train_df,    "[Training Data:] ")
                Stats_Split_Non_Split(validate_df, "[ Test Data   :]")
                print("The Test Samples are: ",randomTestData)


            #************************* Custom generator code ************************
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
            )

            valid_datagen = ImageDataGenerator(
                rescale=1. / 255,
            )

            if(MULTI_CTU_FLAG):
                dictionaryDF = {
                    'Cur_CTU': 'FileName',
                    'N1_CTU': 'N1_CTU',
                    'N2_CTU': 'N2_CTU',
                    'N3_CTU': 'N3_CTU',
                    'N4_CTU': 'N4_CTU',
                    'N1PU': 'N1PU',
                    'N2PU': 'N2PU',
                    'N3PU': 'N3PU',
                    'N4PU': 'N4PU',
                    'N1SL': 'N1SL',
                    'N2SL': 'N2SL',
                    'N3SL': 'N3SL',
                    'N4SL': 'N4SL',
                }
                IMAGE_SIZE_MCTU = (32, 32,1)
                # N1PU N2PU N3PU N4PU N1SL N2SL N3SL N4SL
                inputgenerator = MultiChannelDataGen(train_df,
                                         X_col=dictionaryDF,
                                         y_col={'type': 'CurrSL'},
                                         batch_size=batch_size, input_size=IMAGE_SIZE_MCTU)

                testgenerator = MultiChannelDataGen(validate_df,
                                       X_col=dictionaryDF,
                                       y_col={'type': 'CurrSL'},
                                       batch_size=batch_size, input_size=IMAGE_SIZE_MCTU)
            else:
                inputgenerator = generate_generator_FastCoding(train_datagen, train_df,IMAGE_SIZE,CTU_FLAG,CTU_NN_FLAG,MOTION_FLAG,CTU_NN_Rate_Prev,rate)
                testgenerator = generate_generator_FastCoding(valid_datagen, validate_df,IMAGE_SIZE,CTU_FLAG,CTU_NN_FLAG,MOTION_FLAG,CTU_NN_Rate_Prev,rate)

            # *******************************  [Network Defination]  **********************************
            if (MOTION_FLAG):
                rate_input_shape = (18,)  # have to check
            else:
                rate_input_shape = (9,)

            rate_input = Input(shape=rate_input_shape)
            img_input_shape= (32,32,1)

            # Selection of ML scheme for prediction of non key views

            if (CTU_NN_FLAG):  # CTU + Feature vector case
                z,InpArray=Basic_NN_CTU_FV_D0(img_input_shape,rate_input) # Basic NN CTU + Feature vector case
                    # InpArray=[input_img]
            elif CTU_NN_Rate_Prev:
                rate_input_shape = (1,)
                rate_input = Input(shape=rate_input_shape)
                z, InpArray = Basic_NN_CTU_rate(img_input_shape, rate_input)
            elif (CTU_FLAG):# CTU only case
                z,InpArray=Basic_NN_CTU_D0(img_input_shape) # Basic NN CTU only case
            elif (MULTI_CTU_FLAG):# CTU only case
                img_input_shape= (32,32,5) # The input CTU is 64x64
                z,InpArray=Basic_NN_CTU_D0(img_input_shape) # Basic NN CTU only case
            else:  # only feature vector case
                z,InpArray=NN_FV(rate_input) # NN with Feature vector case


            model = Model(inputs=InpArray, outputs=z)
            model.summary()

            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[
                'acc'])  # , metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...


            class TimeHistory(Callback):
                def on_train_begin(self, logs={}):
                    self.times = []

                def on_epoch_begin(self, epoch, logs={}):
                    self.epoch_time_start = time.time()

                def on_epoch_end(self, epoch, logs={}):
                    self.times.append(time.time() - self.epoch_time_start)


            #time_callback = TimeHistory()

            # Flags used in paper
            # FLAGS_VALUE = 'Rate_' + rate + '_TEST_' + TestInfo + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG)
            # Flags used to add model
            FLAGS_VALUE = 'Rate_' + rate + '_TEST_' + TestInfo + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev)
            print(FLAGS_VALUE)

            modelPath = pathDrive + FLAGS_VALUE + '_model_val_acc_best.h5'


            checkpoint = ModelCheckpoint(filepath=modelPath,
                                         verbose=1, monitor='val_acc',
                                         save_best_only=True, mode='max')

            history = model.fit(inputgenerator, steps_per_epoch=np.floor(len(train_df) / batch_size), epochs=EPOCHS, shuffle=True,
                                callbacks=[checkpoint], verbose=1, validation_data=testgenerator,
                                validation_steps=np.floor(len(validate_df) / batch_size))

            # modelPath = pathDrive + FLAGS_VALUE + 'model_last_epoch.h5'
            # model.save(modelPath)

            hist_df = pd.DataFrame(history.history)
            hist_csv_file = pathDrive + FLAGS_VALUE + 'Summary' + '.csv'
            with open(hist_csv_file, mode='w') as t:
                hist_df.to_csv(t)
            # To HERE
    print(" *******  Depth 1 Code is done ********")

