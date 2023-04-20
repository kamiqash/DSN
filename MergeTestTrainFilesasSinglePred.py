import numpy as np
import pandas as pd
from pathlib import Path
import os

# Author: Kamran Qureshi & Waqas Ahmad, Date: 04-21-23

isColab = False

NonSparseD = [2, 4, 6, 8]
SparseD = [1, 3, 5, 7, 9]
SparseND = np.array(SparseD)
NonSparseND = np.array(NonSparseD)

DEBUG=0
loopidx=1
Record=[]
#-----------------------------------------------------------------------------------------------------------------------
sz1 = 256


SingleFolderDepthWisePred=1

outputDNNFolder= 'FullLF_RateByRate'
outputFolderMergedViews = 'MergedViews_FullLF_RateByRate'
SingleFolderPath='../../output/' + outputDNNFolder + '/' + outputFolderMergedViews + '/'
if not os.path.exists(SingleFolderPath):
    os.makedirs(SingleFolderPath)



#viewsArray=np.array([2,4,6,8,20,22,24,26,38,40,42,44,56,58,60,62,74,76,78,80]) # TwentyViews

# viewsArray=np.array([1,3,5,7,9,19,21,23,25,27,37,39,41,43,45,55,57,59,61,63,73,75,77,79,81]) # TwentyFiveViews
viewsArray=np.array([41]) # Intra Only copy for Full_LF

# -------------------------------- Depth 0 Correction -----------------------------------
DEPTH0 = 1
DEPTH1 = 1
DEPTH2 = 1

rateList=['22','27','32']
TEST_Array=['A','B','C']

BDPSNR=1


if(BDPSNR):
    randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
    randomTestData_27 = randomTestData_22
    randomTestData_32 = randomTestData_22
else:
    randomTestData_22 = [  ['14', '23', '28'],['4', '10', '13'],['5', '7', '16'] ]
    randomTestData_27 = [  ['5', '10', '25'],['6', '12', '26'] ,['15', '19', '22'] ]
    randomTestData_32 = [  ['6', '12', '22'],['9', '17', '19'] ,['3', '6', '12']  ]

for IndexRate in range(0,len(rateList)):
    rate=rateList[IndexRate]
    for IndexTest in range(0,len(TEST_Array)):
        Test = TEST_Array[IndexTest]

        if (rate == '22'):
            randomTestData = randomTestData_22[IndexTest]
        elif (rate == '27'):
            randomTestData = randomTestData_27[IndexTest]
        elif (rate == '32'):
            randomTestData = randomTestData_32[IndexTest]


        if(DEPTH0):
            depthVal = '0'
            TrainFilesPath = '../../DepthWise_labeldata/'
            TestFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_TestImages/'
            PredictionFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_PredictedImages/'
            if not os.path.exists(PredictionFilesPath):
                os.makedirs(PredictionFilesPath)

            for lfname in randomTestData:

                splitCount = 0
                NonsplitCount = 0

                depthVal='0'


                Depth0FileNameTrain = Path(TrainFilesPath + 'HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth0FileNameTest = Path(TestFilesPath + 'Test_'+Test+'_DL_Depth'+depthVal+'_Rate_' + rate + '_LF' + lfname + '.txt')

                if(SingleFolderDepthWisePred):
                    Depth0FileNamePrediction = Path(SingleFolderPath + 'DL_HCI_' + lfname+ '_Rate_' + rate + '_Test_' + Test + '_Depth' + depthVal + '.txt')
                else:
                    Depth0FileNamePrediction = Path(PredictionFilesPath + 'DL_HCI_' + lfname + '_Rate_' + rate + '_Test_' + Test + '_Depth'+depthVal+'.txt')


                Depth0FileNamePredictionSum = Path(PredictionFilesPath + 'DLPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth0FileNamePredictionSumGT = Path(PredictionFilesPath + 'GTPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')

                if not (os.path.exists(Depth0FileNameTest)):
                    print("Warning: This file is not present and hence loop is skipped: ", Depth0FileNameTest)
                    continue

                if not  (os.path.exists(Depth0FileNameTrain)):
                    print("Warning: This file is not present and hence loop is skipped: ", Depth0FileNameTrain)
                    continue


                SKIP_TEST_D0=0
                SKIP_TRAIN_D0=0

                if (os.path.exists(Depth0FileNameTest)):
                    df_Test = pd.read_table(Depth0FileNameTest, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TEST_D0=1


                if (os.path.exists(Depth0FileNameTrain)):

                    df_Train = pd.read_table(Depth0FileNameTrain, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TRAIN_D0=1

                #print(df_Train)
                df_Pred = df_Train.copy()
                count = 0
                PartSum = np.zeros(81)
                PartSumGT= np.zeros(81)

                df_append = pd.DataFrame()
                for viewIndex in range(1, 82):
                    res=(viewsArray == viewIndex)
                    Val64=0
                    if not (res.any()): # If current view belong of 56 non Key views
                        if(SKIP_TEST_D0): # for Twenty views research change if not to if
                            #print("Key View",viewIndex)
                            df_Pred.loc[(viewIndex-1)*64:(viewIndex-1)*64+63]=df_Test.loc[count*64:count*64+63].to_numpy()
                            arr64=df_Test.loc[count*64:count*64+63].to_numpy()
                            Val64 = sum(arr64[:, 9])
                            arr64_GT = df_Train.loc[(viewIndex-1)*64:(viewIndex-1)*64+63].to_numpy()
                            Val64_GT= sum(arr64_GT[:, 9])
                            count = count + 1
                    else:# If current view belong of 25 Key views
                        if(SKIP_TRAIN_D0):
                            #print("Normal View",viewIndex)
                            arr64 = df_Train.loc[(viewIndex - 1) * 64:(viewIndex - 1) * 64 + 63].to_numpy()
                            Val64 = sum(arr64[:, 9])
                            Val64_GT = Val64


                    PartSum[viewIndex-1]=Val64
                    PartSumGT[viewIndex-1]=Val64_GT

                PartSumDF=df = pd.DataFrame(np.array(PartSum))
                PartSumGTDF = df = pd.DataFrame(np.array(PartSumGT))
                np.savetxt(Depth0FileNamePrediction, df_Pred.values,fmt='%d')
                np.savetxt(Depth0FileNamePredictionSum, PartSumDF.values,fmt='%d')
                np.savetxt(Depth0FileNamePredictionSumGT, PartSumGTDF.values, fmt='%d')
                print("Done with saving   ",Depth0FileNamePrediction)

        # -------------------------------- Depth 1 Correction -----------------------------------

        if (DEPTH1):
            depthVal = '1'
            prevdepthVal = '0'

            TrainFilesPath = '../../DepthWise_labeldata/'
            TestFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_TestImages/'
            PredictionFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_PredictedImages/'
            Depth0FilesPath='../../output/'+ outputDNNFolder +'/Depth' + prevdepthVal + '/QP_' + rate + '_PredictedImages/'

            if not os.path.exists(PredictionFilesPath):
                os.makedirs(PredictionFilesPath)

            for lfname in randomTestData:
                splitCount = 0
                NonsplitCount = 0

                Depth1FileNameTrain = Path(TrainFilesPath + 'HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth1FileNameTest = Path(TestFilesPath + 'Test_'+Test+'_DL_Depth'+depthVal+'_Rate_' + rate + '_LF' + lfname + '.txt')


                if(SingleFolderDepthWisePred):
                    Depth1FileNamePrediction = Path(SingleFolderPath + 'DL_HCI_' + lfname + '_Rate_' + rate + '_Test_' + Test + '_Depth' + depthVal + '.txt')
                else:
                    Depth1FileNamePrediction = Path(PredictionFilesPath + 'DL_HCI_' + lfname + '_Rate_' + rate + '_Test_' + Test + '_Depth'+depthVal+'.txt')

                Depth0FileNamePredictionSumD0 = Path(Depth0FilesPath + 'DLPartSum_HCI_' + lfname + '_' + rate + '_Depth'+prevdepthVal+'.txt')
                Depth0FileNamePredictionSumGTD0 = Path(Depth0FilesPath + 'GTPartSum_HCI_' + lfname + '_' + rate + '_Depth'+prevdepthVal+'.txt')

                Depth1FileNamePredictionSumD1 = Path(PredictionFilesPath + 'DLPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth1FileNamePredictionSumGTD1 = Path(PredictionFilesPath + 'GTPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')


                SKIP_TEST_D1=0
                SKIP_TRAIN_D1=0

                if (os.path.exists(Depth1FileNameTest)):
                    df_Test_D1 = pd.read_table(Depth1FileNameTest, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TEST_D1=1


                if (os.path.exists(Depth1FileNameTrain)):

                    df_Train_D1 = pd.read_table(Depth1FileNameTrain, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TRAIN_D1=1

                #--------------  Reading Information from Depth 0 -----------------
                df_Sum_D0 = pd.read_table(Depth0FileNamePredictionSumD0, delimiter=" ", header=None, dtype=np.int32)
                df_Sum_D0_GT = pd.read_table(Depth0FileNamePredictionSumGTD0, delimiter=" ", header=None, dtype=np.int32)


                df_Sum_np_D0=np.array(df_Sum_D0)
                df_Sum_D0_GT=np.array(df_Sum_D0_GT)

                df_Pred_D1 = df_Train_D1.copy()

                count = 0
                PartSum_D1 = np.zeros(81)
                PartSumGT_D1= np.zeros(81)

                BlockStart=0
                BlockEnd = int(int(int(df_Sum_D0_GT[0]) * 4) - 1)
                BlockStart_k=0
                BlockEnd_k=-1

                Depth1List=[]

                for viewIndex in range(1, 82):
                    res = (viewsArray == viewIndex)
                    Val64 = 0

                    if not (res.any()):# If current view belong of 56 non Key views
                        # BlockStart = BlockStart + int(int(int(df_Sum_D0_GT[viewIndex - 1]) * 4))
                        # BlockEnd = BlockEnd + int(int(int(df_Sum_D0_GT[viewIndex]) * 4))
                        if(SKIP_TEST_D1): # for Twenty views research change if not to if
                            # print("Key View", viewIndex)
                            # print([BlockStart, BlockEnd])
                            BlockEnd_k = BlockEnd_k + int(int(int(df_Sum_np_D0[viewIndex-1]) * 4))
                            # print([BlockStart_k, BlockEnd_k])
                            #df_Pred_D1.loc[BlockStart:BlockEnd] = df_Test.loc[BlockStart_k:BlockEnd_k].to_numpy()
                            for idx in range(BlockStart_k,BlockEnd_k+1):
                                 Depth1List.append(df_Test_D1.loc[idx,:])
                            #print("length=", len(Depth1List), "Viewno", viewIndex)
                            count = count + 1
                            arr64 = df_Test_D1.loc[BlockStart_k:BlockEnd_k].to_numpy()
                            Val64 = sum(arr64[:, 9])

                            BlockStart_k = BlockStart_k + int(int(int(df_Sum_np_D0[viewIndex - 1]) * 4))
                        if (SKIP_TRAIN_D1):
                            # Here i am reading Training data view information to calculate number of split decision for that view. It is needed to correctly calculate the shift.
                            arr64_GT = df_Train_D1.loc[BlockStart:BlockEnd].to_numpy()
                            Val64_GT= sum(arr64_GT[:, 9])
                            if(viewIndex<81):
                                BlockStart = BlockStart + int(int(int(df_Sum_D0_GT[viewIndex - 1]) * 4))
                                BlockEnd = BlockEnd + int(int(int(df_Sum_D0_GT[viewIndex]) * 4))

                    else:
                        if (SKIP_TRAIN_D1):
                            #print("Normal View", viewIndex)
                            #print([BlockStart, BlockEnd])

                            for idx in range(BlockStart,BlockEnd+1):
                                 Depth1List.append(df_Train_D1.loc[idx,:])
                            #print("length=",len(Depth1List),"Viewno",viewIndex)

                            arr64 = df_Train_D1.loc[BlockStart:BlockEnd].to_numpy()
                            Val64 = sum(arr64[:, 9])
                            Val64_GT=Val64

                            if(viewIndex<81):
                                BlockStart = BlockStart + int(int(int(df_Sum_D0_GT[viewIndex-1]) * 4))
                                BlockEnd = BlockEnd + int(int(int(df_Sum_D0_GT[viewIndex]) * 4))

                    PartSumGT_D1[viewIndex - 1] = Val64_GT
                    PartSum_D1[viewIndex - 1] = Val64

                    PartSumDF_D1 = df = pd.DataFrame(np.array(PartSum_D1))
                    PartSumDF_D1_GT = df = pd.DataFrame(np.array(PartSumGT_D1))



                df_depth1 = pd.DataFrame(np.array(Depth1List))
                np.savetxt(Depth1FileNamePrediction, df_depth1.values, fmt='%d')
                np.savetxt(Depth1FileNamePredictionSumD1, PartSumDF_D1.values, fmt='%d')
                np.savetxt(Depth1FileNamePredictionSumGTD1, PartSumDF_D1_GT.values, fmt='%d')

                #print("Final length=", len(Depth1List))
                print("Done with saving   ", Depth1FileNamePrediction)

                # -------------------------------- Depth 2 Correction -----------------------------------

        if (DEPTH2):

            depthVal = '2'
            prevdepthVal = '1'

            TrainFilesPath = '../../DepthWise_labeldata/'
            TestFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_TestImages/'
            PredictionFilesPath = '../../output/'+ outputDNNFolder +'/Depth' + depthVal + '/QP_' + rate + '_PredictedImages/'
            Depth1FilesPath='../../output/'+ outputDNNFolder +'/Depth' + prevdepthVal + '/QP_' + rate + '_PredictedImages/'

            if not os.path.exists(PredictionFilesPath):
                os.makedirs(PredictionFilesPath)

            for lfname in randomTestData:
                splitCount = 0
                NonsplitCount = 0

                Depth2FileNameTrain = Path(TrainFilesPath + 'HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth2FileNameTest = Path(TestFilesPath + 'Test_'+Test+'_DL_Depth'+depthVal+'_Rate_' + rate + '_LF' + lfname + '.txt')

                if(SingleFolderDepthWisePred):
                    Depth2FileNamePrediction = Path(SingleFolderPath + 'DL_HCI_' + lfname + '_Rate_' + rate + '_Test_' + Test + '_Depth' + depthVal + '.txt')
                else:
                    Depth2FileNamePrediction = Path(PredictionFilesPath + 'DL_HCI_' + lfname + '_Rate_' + rate + '_Test_' + Test +  '_Depth'+depthVal+'.txt')


                PartitionSumD1_Pred = Path(Depth1FilesPath + 'DLPartSum_HCI_' + lfname + '_' + rate + '_Depth'+prevdepthVal+'.txt')
                PartitionSumD1_GT = Path(Depth1FilesPath + 'GTPartSum_HCI_' + lfname + '_' + rate + '_Depth'+prevdepthVal+'.txt')

                Depth1FileNamePredictionSumD2 = Path(PredictionFilesPath + 'DLPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')
                Depth1FileNamePredictionSumGTD2 = Path(PredictionFilesPath + 'GTPartSum_HCI_' + lfname + '_' + rate + '_Depth'+depthVal+'.txt')

                SKIP_TEST_D2=0
                SKIP_TRAIN_D2=0


                if (os.path.exists(Depth2FileNameTest)):
                    #print("Warning: This file is not present and hence loop is skipped: ", Depth2FileNameTest)
                    df_Test_D2 = pd.read_table(Depth2FileNameTest, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TEST_D2=1
                    #continue

                if (os.path.exists(Depth2FileNameTrain)):
                    #print("Warning: This file is not present and hence loop is skipped: ", Depth2FileNameTest)
                    df_Train_D2 = pd.read_table(Depth2FileNameTrain, delimiter=" ", header=None, dtype=np.int32)
                    SKIP_TRAIN_D2=1
                    #continue

                #df_Test_D2 = pd.read_table(Depth2FileNameTest, delimiter=" ", header=None, dtype=np.int32)
                #df_Train_D2 = pd.read_table(Depth2FileNameTrain, delimiter=" ", header=None, dtype=np.int32)


                # --------------  Reading Information from Depth 1 -----------------
                df_Sum_D1 = pd.read_table(PartitionSumD1_Pred, delimiter=" ", header=None, dtype=np.int32)
                df_Sum_D1_GT = pd.read_table(PartitionSumD1_GT, delimiter=" ", header=None,dtype=np.int32)
                df_Sum_np_D1 = np.array(df_Sum_D1)
                df_Sum_D1_GT = np.array(df_Sum_D1_GT)

                count = 0
                PartSum_D2 = np.zeros(81)
                BlockStart = 0
                BlockEnd = int(int(int(df_Sum_D1_GT[0]) * 4) - 1)
                BlockStart_k = 0
                BlockEnd_k = -1

                Depth2List = []

                for viewIndex in range(1, 82):
                    res = (viewsArray == viewIndex)
                    if not (res.any()):  # for Twenty views research change if not to if
                         if(SKIP_TEST_D2):
                            #print("Key View", viewIndex)
                            #print([BlockStart, BlockEnd])
                            BlockEnd_k = BlockEnd_k + int(int(int(df_Sum_np_D1[viewIndex - 1]) * 4))

                            # print([BlockStart_k, BlockEnd_k])
                            # df_Pred_D1.loc[BlockStart:BlockEnd] = df_Test.loc[BlockStart_k:BlockEnd_k].to_numpy()
                            for idx in range(BlockStart_k, BlockEnd_k + 1):
                                Depth2List.append(df_Test_D2.loc[idx, :])
                            # print("length=", len(Depth1List), "Viewno", viewIndex)

                            BlockStart_k = BlockStart_k + int(int(int(df_Sum_np_D1[viewIndex - 1]) * 4))
                         # Here I have to calculate the block offset in training data corresponding to above test data view.
                         if (SKIP_TRAIN_D2):
                             if(viewIndex<81):
                                BlockStart = BlockStart + int(int(int(df_Sum_D1_GT[viewIndex - 1]) * 4))
                                BlockEnd = BlockEnd + int(int(int(df_Sum_D1_GT[viewIndex]) * 4))
                    else:
                         if (SKIP_TRAIN_D2):
                            #print("Normal View", viewIndex)
                            # print([BlockStart, BlockEnd])
                            for idx in range(BlockStart, BlockEnd + 1):
                                Depth2List.append(df_Train_D2.loc[idx, :])
                            # print("length=",len(Depth1List),"Viewno",viewIndex)
                            if (viewIndex < 81):
                                BlockStart = BlockStart + int(int(int(df_Sum_D1_GT[viewIndex - 1]) * 4))
                                BlockEnd = BlockEnd + int(int(int(df_Sum_D1_GT[viewIndex]) * 4))

                # print("Final length=", len(Depth2List))

                df_depth2 = pd.DataFrame(np.array(Depth2List))
                np.savetxt(Depth2FileNamePrediction, df_depth2.values, fmt='%d')
                print("Done with saving   ", Depth2FileNamePrediction)





