# This Function is called by function main_Predictor and gets model and test LF images need to be evaluated using ML scheme for Depth 0.
# The ML scheme output is stored in the text file specified by the variable filename2Store
# Author: Waqas Ahmad, Date: 25-02-2022

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from csv import writer
import numpy as np
import tensorflow as tf
import sys
import os

def writeDepth0TestPrediction(model,Test_df,Test,rate, depthValue,TestImage,outputDNNFolder,methodology,CTU_NN_FLAG,CTU_FLAG,CTU_NN_Rate_Prev):
    # methodology=1
    # 13-04-23: Changing to from output to output_test since testing F1-score
    pathDrive = '../../output/' + outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(rate) + '_TestImages/'
    if not os.path.exists(pathDrive):
        os.makedirs(pathDrive)

    depthVal=0
    npList = []
    nameList = []
    vecList = []

    nameFile = "Test_%s_DL_Depth%d_Rate_%d_LF%d.txt" % (Test, depthVal, int(rate), int(TestImage))

    # Code to read test images and predict using trained DL network
    NameofTestImage = Test_df['FileName']
    CTU_L = Test_df['CTUno']
    LF_L = Test_df['LFname']
    PU_L = Test_df['CurrPU']
    VN = Test_df['viewNumber']

    N1PU_ARR = np.asarray(Test_df['N1PU'])
    N2PU_ARR = np.asarray(Test_df['N2PU'])
    N3PU_ARR = np.asarray(Test_df['N3PU'])
    N4PU_ARR = np.asarray(Test_df['N4PU'])

    N1SL_ARR = np.asarray(Test_df['N1SL'])
    N2SL_ARR = np.asarray(Test_df['N2SL'])
    N3SL_ARR = np.asarray(Test_df['N3SL'])
    N4SL_ARR = np.asarray(Test_df['N4SL'])

    rateValue_ARR = np.asarray(Test_df['rate'])
    filename2Store = pathDrive + nameFile
    # print(filename2Store)
    # myfile = Path(filename2Store)
    # myfile.touch(exist_ok=True)
    for ImgIdx in range(0, len(NameofTestImage)):
        # for ent in Test_df['FileName']:
        ent = NameofTestImage[ImgIdx]
        image3 = tf.keras.preprocessing.image.load_img(ent, color_mode='grayscale')
        image_arr3 = tf.keras.preprocessing.image.img_to_array(image3)
        image_arr3 = tf.image.resize(image_arr3, (64, 64)).numpy()
        image_arr3 = image_arr3 / 255.0
        npList.append(image_arr3)
        nameList.append(ent)
    npList = np.asarray(npList)

    n1pu = np.array(N1PU_ARR).astype(float)
    n2pu = np.array(N2PU_ARR).astype(float)
    n3pu = np.array(N3PU_ARR).astype(float)
    n4pu = np.array(N4PU_ARR).astype(float)

    n1sl = np.array(N1SL_ARR).astype(float)
    n2sl = np.array(N2SL_ARR).astype(float)
    n3sl = np.array(N3SL_ARR).astype(float)
    n4sl = np.array(N4SL_ARR).astype(float)
    # 12/01/21 checking rate
    rateValue = np.array(rateValue_ARR).astype(float)
    rateValue = (rateValue - 8)/ (45 - 8)

    # print(np.shape(npList))
    # print(np.shape(vectorTuple))
    # ,CTU_NN_FLAG,CTU_FLAG,CTU_NN_Rate_Prev
    if CTU_NN_FLAG: #CTU+Feature
        #     Features
        vectorTuple = np.asarray(tf.transpose(tuple([n1pu, n2pu, n3pu, n4pu, n1sl, n2sl, n3sl, n4sl,rateValue])))
        # CTU = npList
        Input = tuple([npList, vectorTuple])
        p = model.predict(Input)
    elif CTU_FLAG: #Only CTU
        # CTU = npList
        xInput_fullLF = npList
        p = model.predict(xInput_fullLF)
    elif CTU_NN_Rate_Prev: #CTU + Rate + Prev Layer Model
        # Input for CNN + Rate + PrevLayer Model
        #CTU = npList && rateValue = rate
        Input = tuple([npList, rateValue])
        p = model.predict(Input)
    # if methodology == 1:
    #     p = model.predict(Input)
    #     #
    #
    # else:
    #     p = model.predict(xInput)
    p[p > 0.5] = 1
    p[p <= 0.5] = 0
    # print(p)
    score = accuracy_score(Test_df['CurrSL'].astype(int), p)
    print('F1 is: ', f1_score(Test_df['CurrSL'].astype(int), p))
    print('Precision is: ', precision_score(Test_df['CurrSL'].astype(int), p))
    print('Recall is: ', recall_score(Test_df['CurrSL'].astype(int), p))
    F1Score = f1_score(Test_df['CurrSL'].astype(int), p)
    Precision = precision_score(Test_df['CurrSL'].astype(int), p)
    Recall = recall_score(Test_df['CurrSL'].astype(int), p)
    List = [Precision,Recall,F1Score]
    tn, fp, fn, tp = confusion_matrix(Test_df['CurrSL'].astype(int), p, labels=[0,1]).ravel()
    print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ',
          tp)
    ConfusionMatrix = [tn,fp,fn,tp]
    # Path excel sheet paper
    # PathExcelSheet = '../../output/' + outputDNNFolder + '/Depth0/QP_' + str(rate) + '/Rate_' + rate + '_Test_' + Test + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + 'Summary.csv'
    # Path excel sheet add new model
    PathExcelSheet = '../../output/' + outputDNNFolder + '/Depth' + depthValue + '/QP_' + str(
        rate) + '/Rate_' + rate + '_Test_' + Test + '_FLAGS_' + str(CTU_FLAG) + '_' + str(CTU_NN_FLAG) + '_' + str(CTU_NN_Rate_Prev) + 'ConfusionMatrix.csv'
    with open(PathExcelSheet, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(["Precision","Recall","F1-Score"])
        writer_object.writerow(List)
        writer_object.writerow(["True negatives", "False positives", "False negatives","True Positives"])
        writer_object.writerow(ConfusionMatrix)
        f_object.close()
    with open(filename2Store, 'w') as f:
        for pIndex in range(0, len(NameofTestImage)):
            strW = "%d %d 0 0 %d 9 9 9 9 %d\n" % (int(VN[pIndex]), int(CTU_L[pIndex]), int(PU_L[pIndex]), p[pIndex])
            f.write(strW)
            npList = []
        f.close()
        print('File writing at Depth 0, Test image: ' +  TestImage + ' is completed ')