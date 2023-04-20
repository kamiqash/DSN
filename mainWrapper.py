# Branch Combine bitrate, HCI+Inrai DB, Random Test
# This file contain the main function of the framework
# This function is the wrapper that runs First part of Framework which includes Loading of Datasets(depth 0,1 and 2)
# Then it evaluate different variance of the proposed ML Scheme. The trained network and summayr is stored in the specifed folder.
# Author: Waqas ahmad and Kamran Qureshi
# Date: 20-04-2022

#from PIL import Image as pil_image
# Reading function define in other python file
from CNN_Depth0 import Depth0_CNN
from CNN_Depth1 import Depth1_CNN
from CNN_Depth2 import Depth2_CNN


import tensorflow as tf
tf.__version__

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Flags that define different vaiant of the code
METHODOLOGY=1 # 0 for non key views prediction and 1 for complete LF view prediction
MOTION_FLAG=0 # When this flag is 1 then four neighbours motion vector information will be used in feature vector

DEBUG=0 # This flag enable debugging. Total LF are 4 and total test images are 1.

# Define the current System on which the code is running. Then accordingly it defines the path of the datasets
SYSTEM= 2  # 1 Ubuntu System, 2 Windows system and 3 Colab System
# These two flag control following variation of the code (1,0) select CU only case , (0,1) selects CTU + feature vector case, (0,0) select feature only case
# CTU_FLAG_HV    = [1, 0, 0]
# CTU_NN_FLAG_HV = [0, 1, 0]
# CTU_FLAG_HV    = [1, 0]
# CTU_NN_FLAG_HV = [0, 1]
CTU_FLAG_HV    = [1]
CTU_NN_FLAG_HV = [0]
CTU_NN_Rate_Prev = 0
MULTI_CTU_FLAG= 0
kamran = 9
# Select the databaes
# DBlist=['HCI','Inria']
DBlist=['HCI']

PreDefinedTest = 1 # 0 will randomoly select three test cases in which 3 LF images will be used for testing and rest of the LF images will go for training. 1 value will select pre defined test cases

# Bitrates on which experimentation is performed
# rate_list = ['32', '27', '22']
rate_list = ['22']
PRINT_STATUS=1
PRINT_STATS=1
# Test I am using to benchmark the ML schemes. Each test contains 3 LF images which are defined in the inner functions.
# Test = ['A', 'B', 'C']
Test = ['C']
simCount=0
EPOCHS=3
depthValues=['2']
# folder for paper results
# outputDNNFolder= 'CodeforUpload_FULLLF'
# Testing output
# outputDNNFolder= 'TestingNewModel'
# outputDNNFolder= 'DNN_For_BDPSNR_Validation'
# DNN_Rate_by_Rate #Used for validating DNN_15_02_2022
outputDNNFolder= 'FullLF_RateByRate'
outputFolder='../../output/'+ outputDNNFolder + '/'

for depth in depthValues: # Loop over depth 0 to 2
        for FLAG_indx in range(0,len(CTU_FLAG_HV)): # Loop over flags
            CTU_FLAG = CTU_FLAG_HV[FLAG_indx]         #CTU only
            CTU_NN_FLAG = CTU_NN_FLAG_HV[FLAG_indx]   #CTU + Feature Vector
            for loopidx in range (0,len(Test)):# Loop over Test A to C
                simCount=simCount+1
                print("Simulation no. ",simCount)
                if(depth=='0'): # Depth 0 work
                    print('Depth0')
                    Depth0_CNN(loopidx,CTU_FLAG,MOTION_FLAG,CTU_NN_FLAG,rate_list,DBlist,PRINT_STATUS,PRINT_STATS,Test[loopidx],EPOCHS,outputFolder,SYSTEM,MULTI_CTU_FLAG,METHODOLOGY,PreDefinedTest,DEBUG,CTU_NN_Rate_Prev)
                elif(depth=='1'):# Depth 1 work
                    print('Depth1')
                    Depth1_CNN(loopidx,CTU_FLAG,MOTION_FLAG,CTU_NN_FLAG,rate_list,DBlist,PRINT_STATUS,PRINT_STATS,Test[loopidx],EPOCHS,outputFolder,SYSTEM,MULTI_CTU_FLAG,METHODOLOGY,PreDefinedTest,DEBUG,CTU_NN_Rate_Prev)
                else:# Depth 2 work
                     print('Depth 2')
                     Depth2_CNN(loopidx, CTU_FLAG, MOTION_FLAG, CTU_NN_FLAG, rate_list,DBlist, PRINT_STATUS, PRINT_STATS,Test[loopidx], EPOCHS,outputFolder,SYSTEM,MULTI_CTU_FLAG,METHODOLOGY,PreDefinedTest,DEBUG,CTU_NN_Rate_Prev)



