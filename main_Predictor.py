import tensorflow as tf
import sys
from tensorflow import keras
import os
from LoadTestImages import LoadFiftySixViewsDepth0TestImages
from LoadTestImages import LoadFiftySixViewsDepth1TestImages
from LoadTestImages import LoadFiftySixViewsDepth2TestImages

from LoadLFImages_Backup import LoadFiftySixViewsDepth0
from LoadLFImages_Backup import LoadFiftySixViewsDepth1
from LoadLFImages_Backup import LoadFiftySixViewsDepth2

from depth0Writter import writeDepth0TestPrediction
from depth1Writter import writeDepth1TestPrediction
from depth2Writter import writeDepth2TestPrediction


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)
DB=['HCI']
# depthValues=['2']
depthValues=['0','1','2']
methodology=1

CTU_FLAG=1
CTU_NN_FLAG=0


for depthValue in depthValues:
    # simulation terminated by windows
    TEST=['A','B','C']
    # TEST = ['B']
    rateList=['22','27','32']
    # rateList = ['32']
    # Folder path where models are saved
    outputDNNFolder = 'FullLF_RateByRate'

    PRINT_STATUS=1
    isColab=False
    isUbuntu=True
    isBigPCWIN=False

    BDPSNR=1

    if(BDPSNR):
        randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
        randomTestData_27 = randomTestData_22
        randomTestData_32 = randomTestData_22
    else:
        randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
        randomTestData_27 = [  ['5', '10', '25'],['6', '12', '26'] ,['15', '19', '22'] ]
        randomTestData_32 = [  ['6', '12', '22'],['9', '17', '19'] ,['3', '6', '12']  ]

    if(depthValue=='0'):
        for rateIdx in range(0,len(rateList)):
            rate=rateList[rateIdx]

            metaPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/' + rate + '_META/'
            ctuPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/CTU/'

            for TestIdx in range(0,len(TEST)):
                if TEST[TestIdx] == 'A':
                    index = 0
                elif TEST[TestIdx]  == 'B':
                    index = 1
                elif TEST[TestIdx]  == 'C':
                    index = 2
                if (rate == '22'):
                    randomTestData = randomTestData_22[index] #['14', '23','28']
                elif (rate == '27'):
                    randomTestData = randomTestData_27[index]
                elif (rate == '32'):
                    randomTestData = randomTestData_32[index]
                # Model Loading Depth0
                pathDrive = '../../output/'+ outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '/'

                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'

                model = keras.models.load_model(modelPath)
                print(model.summary())
                for ImageIndex in range(0,3):
                    Test_df=LoadFiftySixViewsDepth0TestImages(randomTestData[ImageIndex],ctuPath,metaPath,rate,PRINT_STATUS,methodology)
                    print("No. of Test Samples: ", len(Test_df))
                    writeDepth0TestPrediction(model, Test_df, TEST[TestIdx], rate, depthValue,randomTestData[ImageIndex],outputDNNFolder,CTU_NN_FLAG,CTU_FLAG)

    elif(depthValue=='1'):
        for rateIdx in range(0,len(rateList)):
            rate=rateList[rateIdx]
            metaPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/META_' + rate + '/'
            ctuPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/CTU_' + rate + '/'
            metaPath_Test = 'C:\PHD\Dataset2022\Depth' + depthValue + '/META_' + '100' + '/'
            ctuPath_Test = 'C:\PHD\Dataset2022\Depth' + depthValue + '/CTU_' + '100' + '/'
            for TestIdx in range(0, len(TEST)):
                if TEST[TestIdx]  == 'A':
                    index = 0
                elif TEST[TestIdx]  == 'B':
                    index = 1
                elif TEST[TestIdx]  == 'C':
                    index = 2
                if (rate == '22'):
                    randomTestData = randomTestData_22[index]
                elif (rate == '27'):
                    randomTestData = randomTestData_27[index]
                elif (rate == '32'):
                    randomTestData = randomTestData_32[index]
                # Model Loading Depth1
                pathDrive = '../../output/' + outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '/'

                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'

                model = keras.models.load_model(modelPath)
                print(model.summary())

                for ImageIndex in range(0,len(randomTestData)):
                    Test_df = LoadFiftySixViewsDepth1TestImages(randomTestData[ImageIndex], ctuPath, metaPath,
                                                                metaPath_Test, ctuPath_Test, rate, PRINT_STATUS,
                                                                TEST[TestIdx], randomTestData[ImageIndex],outputDNNFolder,methodology)
                    if (len(Test_df)==0):
                        print("The array is empty. There is no CU which split in depth 1")
                    else:
                        print("No. of Test Samples: ", len(Test_df))
                        writeDepth1TestPrediction(model, Test_df, TEST[TestIdx], rate, depthValue, randomTestData[ImageIndex],outputDNNFolder,CTU_NN_FLAG,CTU_FLAG)

    elif(depthValue=='2'):
        for rateIdx in range(0, len(rateList)):
            rate=rateList[rateIdx]

            metaPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/META_' + rate + '/'
            ctuPath = 'C:\PHD\Dataset2022\Depth' + depthValue + '/CTU_' + rate + '/'
            metaPath_Test = 'C:\PHD\Dataset2022\Depth' + depthValue + '/META_' + '200' + '/'
            ctuPath_Test = 'C:\PHD\Dataset2022\Depth' + depthValue + '/CTU_' + '200' + '/'
            # Model Loading Depth 2
            for TestIdx in range(0, len(TEST)):
                if TEST[TestIdx]  == 'A':
                    index = 0
                elif TEST[TestIdx]  == 'B':
                    index = 1
                elif TEST[TestIdx]  == 'C':
                    index = 2
                if (rate == '22'):
                    randomTestData = randomTestData_22[index]
                elif (rate == '27'):
                    randomTestData = randomTestData_27[index]
                elif (rate == '32'):
                    randomTestData = randomTestData_32[index]

                pathDrive = '../../output/'+ outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '/'

                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'
                model = keras.models.load_model(modelPath)
                print(model.summary())

                for ImageIndex in range(0,len(randomTestData)):

                    # ------------------------------- [Reading Depth1 Information] --------------------------------------
                    Depth1pathDrive = '../../output/' + outputDNNFolder + '/Depth1/QP_' + str(rate) + '_TestImages/'
                    Depth1FileName = "Test_%s_DL_Depth%d_Rate_%d_LF%d.txt" % (TEST[TestIdx], 1, int(rate), int(randomTestData[ImageIndex]))
                    Depth1file = Depth1pathDrive + Depth1FileName
                    if not os.path.exists(Depth1file):
                        print("Sorry File Does not Exist ",Depth1FileName)
                        continue
                    # ---------------------------------------------------------------------------------------------------
                    print("Reading Depth 1 File: %s", Depth1FileName)
                    Test_df=LoadFiftySixViewsDepth2TestImages(randomTestData[ImageIndex],ctuPath,metaPath,metaPath_Test,ctuPath_Test,rate,PRINT_STATUS,Depth1file,methodology)

                    if (len(Test_df)==0):
                        print("The array is empty. There is no CU which split in depth 2")
                    else:
                        print("No. of Test Samples: ", len(Test_df))
                        writeDepth2TestPrediction(model, Test_df, TEST[TestIdx], rate,depthValue, randomTestData[ImageIndex],outputDNNFolder,CTU_NN_FLAG,CTU_FLAG)
