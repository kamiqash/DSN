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
# depthValues=['0']
depthValues=['0','1','2']
methodology=1

CTU_FLAG=1
CTU_NN_FLAG=0
CTU_NN_Rate_Prev=0

for depthValue in depthValues:
    # simulation terminated by windows
    TEST=['A','B','C']
    # TEST = ['B']
    rateList=['22','27','32']
    # rateList = ['32']
    # folder used for paper
    # outputDNNFolder= 'CodeforUpload_FULLLF'
    # folder used to add new model
    # TestingNewModel #New model created where rate + prev layer parameters are added
    # DNN_Rate_by_Rate $Validation of DNN_15_02_2022 working
    # FullLF_RateByRate
    outputDNNFolder = 'FullLF_RateByRate'
    # outputDNNFolder='DNN_For_BDPSNR'
    PRINT_STATUS=1
    isColab=False
    isUbuntu=True
    isBigPCWIN=False

    ##FLAGS_VALUE= 'Rate_'+ rate + '_TEST_'+ TestInfo + '_FLAGS_' + str(CTU_FLAG) +'_'+  str(CTU_NN_FLAG) +'_'+ str(KIM_CNN) +'_'+ str(FLAG_20) + '_'
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

            # metaPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/' + rate + '_META/'
            # ctuPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/CTU/'
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
                # 13-04-23: Changing to from output to output_test since testing F1-score
                # + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev)
                pathDrive = '../../output/'+ outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '/'
                # Flags used in paper
                # modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'
                # modelPath = pathDrive + 'Rate_Combined' + '_TEST_' + TEST[TestIdx] + '_FLAGS' + '_1_0_' + 'model_val_acc_best.h5'
                # Flags used for adding new model
                # Rate_22_TEST_A_FLAGS_0_1_0_model_val_acc_best
                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev) + '_model_val_acc_best.h5'
                # Rate_Combined_TEST_A_FLAGS_1_0_model_val_acc_best
                # Path for All LF CTU
                # modelPath = pathDrive + 'Rate_Combined' + '_TEST_' + TEST[TestIdx] + '_FLAGS' + '_1_0_' + 'model_val_acc_best.h5'
                model = keras.models.load_model(modelPath)
                print(model.summary())
                for ImageIndex in range(0,3):
                    # Test_df = LoadFiftySixViewsDepth0(randomTestData[ImageIndex], ctuPath, metaPath, rate,PRINT_STATUS,methodology,DB[0])
                    Test_df=LoadFiftySixViewsDepth0TestImages(randomTestData[ImageIndex],ctuPath,metaPath,rate,PRINT_STATUS,methodology)
                    print("No. of Test Samples: ", len(Test_df))
                    writeDepth0TestPrediction(model, Test_df, TEST[TestIdx], rate, depthValue,randomTestData[ImageIndex],outputDNNFolder,methodology,CTU_NN_FLAG,CTU_FLAG,CTU_NN_Rate_Prev)

    elif(depthValue=='1'):
        for rateIdx in range(0,len(rateList)):
            rate=rateList[rateIdx]
            # randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
            # metaPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/META_' + rate + '/'
            # ctuPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/CTU_' + rate + '/'
            # metaPath_Test = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/META_' + '100' + '/'
            # ctuPath_Test = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/CTU_' + '100' + '/'
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
                # 13-04-23: Changing to from output to output_test since testing F1-score
                pathDrive = '../../output/' + outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '/'
                # modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS' + '_0_1_0_0__' + 'model_val_acc_best.h5'
                # Rate_22_TEST_A_FLAGS_1_0_model_val_acc_best
                # Path for CTU FULL LF model
                # Path Used in Paper
                # modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'
                # Flags used for adding new model
                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev) + '_model_val_acc_best.h5'
                # D:\output\CodeforUpload_FULLLF\Depth1\QP_22
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
                        writeDepth1TestPrediction(model, Test_df, TEST[TestIdx], rate, depthValue, randomTestData[ImageIndex],outputDNNFolder,methodology,CTU_NN_FLAG,CTU_FLAG,CTU_NN_Rate_Prev)

    elif(depthValue=='2'):
        for rateIdx in range(0, len(rateList)):
            rate=rateList[rateIdx]
            # randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
            # metaPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/META_' + rate + '/'
            # ctuPath = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/CTU_' + rate + '/'
            # metaPath_Test = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/META_' + '200' + '/'
            # ctuPath_Test = '/home/realistic3d/WaqasAH/FastCodingWork/Dataset_HCI/Depth' + depthValue + '/CTU_' + '200' + '/'
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
                # modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS' + '_0_1_0_0__' + 'model_val_acc_best.h5'
                # MODEL PATH FOR FULL CTU LF
                # updated 13 Jan 23
                # Flags Used in paper
                # modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_model_val_acc_best.h5'
                # Flags used for adding new model
                modelPath = pathDrive + 'Rate_' + rate + '_TEST_' + TEST[TestIdx] + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev) + '_model_val_acc_best.h5'
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
                        writeDepth2TestPrediction(model, Test_df, TEST[TestIdx], rate,depthValue, randomTestData[ImageIndex],outputDNNFolder,methodology,CTU_NN_FLAG,CTU_FLAG,CTU_NN_Rate_Prev)
