# This Function is called by function main_Predictor and it gets path of Datasets. This function loads the dataset and return the information in panda frame.
# This function is used for all depths 0 to 2
# Author: Waqas Ahmad and Kamran Qureshi, Date: 25-02-2022

import numpy as np
import pandas as pd
import os
import math

#--------------------------------------- 56 Views Work Test Images -----------------------------------------------------
# Depth0 reading
def LoadFiftySixViewsDepth0TestImages(LFImages,ctuPath,metaPath,rate,PRINT_STATUS,METHODOLOGY):
    Record=[]
    sz1 = 256
    sz2 = sz1 * 2
    MVX_1 = np.zeros(sz2)
    MVX_2 = np.zeros(sz2)
    MVY_1 = np.zeros(sz2)
    MVY_2 = np.zeros(sz2)
    MVX_Arr = []

    NonSparseD = [2, 4, 6, 8]
    SparseD = [1, 3, 5, 7, 9]
    SparseND = np.array(SparseD)
    NonSparseND = np.array(NonSparseD)

    lfname =LFImages
    if (PRINT_STATUS):
        print("Loading LF ", lfname)
    for lfC in range(1, 10):
        for lfR in range(1, 10):
            viewNumber = (lfC - 1) * 9 + (lfR - 1) + 1
            #print("[%d,%d] %d", lfR, lfC, viewNumber)
            for CTUno in range(0, 64):
                CurrPU = 0
                CurrSL = 0
                pufilenameCurrInfo = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                    lfR) + '_C' + str(
                    lfC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                pufilenameCurrInfo_CTU = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                    lfC) + '_ctuno' + str(CTUno) + '_d0' + '.png'

                pufilename1_ctu = pufilenameCurrInfo_CTU
                pufilename2_ctu = pufilenameCurrInfo_CTU
                pufilename3_ctu = pufilenameCurrInfo_CTU
                pufilename4_ctu = pufilenameCurrInfo_CTU

                with open(pufilenameCurrInfo) as fpucurr:
                    CurrSL = (int(fpucurr.readline()))
                    CurrPU = (int(fpucurr.readline()))  # / 1
                    MVx1 = (int(fpucurr.readline())) + sz1
                    MVy1 = (int(fpucurr.readline())) + sz1
                    MVx2 = (int(fpucurr.readline())) + sz1
                    MVy2 = (int(fpucurr.readline())) + sz1

                testSR = (lfR == SparseND)
                testNonR = (lfR == NonSparseND)
                testSC = (lfC == SparseND)
                testNonC = (lfC == NonSparseND)

                # Defining Neighbours
                N1PU = 0
                N2PU = 0
                N3PU = 0
                N4PU = 0
                N1SL = 0
                N2SL = 0
                N3SL = 0
                N4SL = 0

                if (METHODOLOGY):  # IF we need to predict complete LF views
                    Record.append(
                        [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                         N4PU,
                         N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,
                         viewNumber])
                # print("Waqas Test")
                # Getting only 5x5 key views
                else:
                    if (testSR.any() and testSC.any()):
                        #print("DCV",lfR,lfC)
                        t = 10
                    elif (testNonR.any() and testNonC.any()):
                        #print("case Sparse", lfR, lfC)
                        N_Ver_LC = lfC - 1
                        N_Ver_LR = lfR - 1
                        N_Ver_RC = lfC + 1
                        N_Ver_RR = lfR + 1

                        N_Ver_LC2 = lfC + 1
                        N_Ver_LR2 = lfR - 1

                        N_Ver_RC2 = lfC - 1
                        N_Ver_RR2 = lfR + 1

                        #print("KSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber), "NL (%d,%d)" % (N_Ver_LR, N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR, N_Ver_RC),"NL2 (%d,%d)" % (N_Ver_LR2, N_Ver_LC2),"NR2 (%d,%d)" % (N_Ver_RR2, N_Ver_RC2))

                        pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.png'

                        with open(pufilename1) as fpu1:
                            N1SL = (int(fpu1.readline()))
                            N1PU = (int(fpu1.readline())) / 10
                            # MVx1_N1 = (int(fpu1.readline()))
                            # MVy1_N1 = (int(fpu1.readline()))
                            # MVx2_N1 = (int(fpu1.readline()))
                            # MVy2_N1 = (int(fpu1.readline()))

                        pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename2)
                        with open(pufilename2) as fpu2:
                            N2SL = (int(fpu2.readline()))
                            N2PU = (int(fpu2.readline())) / 10
                            # MVx1_N2 = (int(fpu2.readline()))
                            # MVy1_N2 = (int(fpu2.readline()))
                            # MVx2_N2 = (int(fpu2.readline()))
                            # MVy2_N2 = (int(fpu2.readline()))

                        pufilename3 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_LR2) + '_C' + str(
                            N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename3_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR2) + '_C' + str(
                            N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename3)
                        with open(pufilename3) as fpu3:
                            N3SL = (int(fpu3.readline()))
                            N3PU = (int(fpu3.readline())) / 10
                            # MVx1_N3 = (int(fpu3.readline()))
                            # MVy1_N3 = (int(fpu3.readline()))
                            # MVx2_N3 = (int(fpu3.readline()))
                            # MVy2_N3 = (int(fpu3.readline()))

                        pufilename4 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_RR2) + '_C' + str(
                            N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename4_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR2) + '_C' + str(
                            N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename4)
                        with open(pufilename4) as fpu4:
                            N4SL = (int(fpu4.readline()))
                            N4PU = (int(fpu4.readline())) / 10
                            # MVx1_N4 = (int(fpu4.readline()))
                            # MVy1_N4 = (int(fpu4.readline()))
                            # MVx2_N4 = (int(fpu4.readline()))
                            # MVy2_N4 = (int(fpu4.readline()))

                        # Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                        #          N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu,
                        #          pufilename4_ctu,pufilenameCurrInfo_CTU])


                        #print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 + ' N3: ' + pufilename3 + ' N4: ' + pufilename4)


                        Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,N4PU,
                             N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber])

                        # Record.append(
                        #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
                        #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,
                        #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
                        #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])

                    elif (testSR.any() and not testSC.any()):
                        N_Ver_LC = lfC - 1
                        N_Ver_LR = lfR
                        N_Ver_RC = lfC + 1
                        N_Ver_RR = lfR
                        pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename1)
                        with open(pufilename1) as fpu1:
                            N1SL = (int(fpu1.readline()))
                            N1PU = (int(fpu1.readline())) / 10
                            MVx1_N1 = (int(fpu1.readline()))
                            MVy1_N1 = (int(fpu1.readline()))
                            MVx2_N1 = (int(fpu1.readline()))
                            MVy2_N1 = (int(fpu1.readline()))

                        pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename2)
                        with open(pufilename2) as fpu2:
                            N2SL = (int(fpu2.readline()))
                            N2PU = (int(fpu2.readline())) / 10
                            MVx1_N2 = (int(fpu2.readline()))
                            MVy1_N2 = (int(fpu2.readline()))
                            MVx2_N2 = (int(fpu2.readline()))
                            MVy2_N2 = (int(fpu2.readline()))
                        N3PU = N1PU
                        N3SL = N1SL
                        N4PU = N2PU
                        N4SL = N2SL
                        #print("SSV [Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                        #print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 )

                        Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,N4PU,
                             N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber])

                    elif (testSC.any() and not testSR.any()):
                        N_Ver_LC = lfC
                        N_Ver_LR = lfR - 1
                        N_Ver_RC = lfC
                        N_Ver_RR = lfR + 1
                        pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                            N_Ver_LC) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename1)
                        with open(pufilename1) as fpu1:
                            N1SL = (int(fpu1.readline()))
                            N1PU = (int(fpu1.readline())) / 10
                            MVx1_N1 = (int(fpu1.readline()))
                            MVy1_N1 = (int(fpu1.readline()))
                            MVx2_N1 = (int(fpu1.readline()))
                            MVy2_N1 = (int(fpu1.readline()))

                        pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_' + rate + '_R' + str(
                            N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.txt'
                        pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                            N_Ver_RC) + '_ctuno' + str(CTUno) + '_d0' + '.png'
                        # print(pufilename2)
                        with open(pufilename2) as fpu2:
                            N2SL = (int(fpu2.readline()))
                            N2PU = (int(fpu2.readline())) / 10
                            MVx1_N2 = (int(fpu2.readline()))
                            MVy1_N2 = (int(fpu2.readline()))
                            MVx2_N2 = (int(fpu2.readline()))
                            MVy2_N2 = (int(fpu2.readline()))

                        N3PU = N1PU
                        N3SL = N1SL
                        N4PU = N2PU
                        N4SL = N2SL

                        MVx1_N3 = MVx1_N1
                        MVy1_N3 = MVy1_N1
                        MVx2_N3 = MVx2_N1
                        MVy2_N3 = MVy2_N1

                        MVx1_N4 = MVx1_N2
                        MVy1_N4 = MVy1_N2
                        MVx2_N4 = MVx2_N2
                        MVy2_N4 = MVy2_N2

                        #print("SSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                       #print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2)

                        Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,N4PU,
                             N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber])

                        # Record.append(
                        #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
                        #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename4_ctu, pufilename4_ctu,
                        #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
                        #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])

    df = pd.DataFrame(np.array(Record),
                  columns=['FileName', 'LFname', 'lfR', 'lfC', 'rate', 'CTUno', 'CurrPU', 'CurrSL', 'N1PU',
                           'N2PU', 'N3PU', 'N4PU', 'N1SL', 'N2SL', 'N3SL','N4SL', 'N1_CTU', 'N2_CTU','N3_CTU','N4_CTU','viewNumber'])
    return df

## Detph 1 reading
def LoadFiftySixViewsDepth1TestImages(LFImages,ctuPath,metaPath,metaPath_Test,ctuPath_Test,rate,PRINT_STATUS,Test,TestImage,outputDNNFolder,METHODOLOGY):

    Record_Depth1=[]
    sz1 = 256
    sz2 = sz1 * 2
    MVX_1 = np.zeros(sz2)
    MVX_2 = np.zeros(sz2)
    MVY_1 = np.zeros(sz2)
    MVY_2 = np.zeros(sz2)
    MVX_Arr = []

    NonSparseD = [2, 4, 6, 8]
    SparseD = [1, 3, 5, 7, 9]
    SparseND = np.array(SparseD)
    NonSparseND = np.array(NonSparseD)

    depthVal=1

    #------------------------------- [Reading Depth0 Information] --------------------------------------
    Depth0pathDrive = '../../output/' + outputDNNFolder + '/Depth0/QP_' + str(rate) + '_TestImages/'
    Depth0FileName = "Test_%s_DL_Depth%d_Rate_%d_LF%d.txt" % (Test, 0, int(rate), int(TestImage))
    Depth0file=Depth0pathDrive + Depth0FileName
    #---------------------------------------------------------------------------------------------------
    print("Reading Depth 0 File: %s",Depth0FileName)

    with open(Depth0file) as fD0:
        lines = fD0.readlines()
        for line in lines:
            Tline = line.split()
            viewNumber = int(Tline[0])
            lfC = math.floor((viewNumber - 1) / 9) + 1
            lfR = (viewNumber - 1) % 9 + 1
            CTUno = int(Tline[1])

            lfname = LFImages

            # if (PRINT_STATUS):
            #     print("Loading LF ", LFImages)

            if (Tline[9] == '1'): # Select only those Samples which were splitted in previous depth

                for PartNo in range(0, 4):
                    CurrPU = 0
                    CurrSL = 0
                    # Dummy Values loading
                    pufilenameCurrInfo = metaPath_Test + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                        lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                    pufilenameCurrInfo_CTU = ctuPath_Test + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                        lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'
                    # GT Values loaded
                    pufilenameCurrInfo_GT = metaPath + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                        lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                    pufilenameCurrInfo_CTU_GT = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                        lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                    pufilename1_ctu = pufilenameCurrInfo_CTU
                    pufilename2_ctu = pufilenameCurrInfo_CTU
                    pufilename3_ctu = pufilenameCurrInfo_CTU
                    pufilename4_ctu = pufilenameCurrInfo_CTU

                    if (os.path.exists(pufilenameCurrInfo)):
                        if (os.path.exists(pufilenameCurrInfo_GT)):
                            with open(pufilenameCurrInfo_GT) as fpucurr:
                                CurrSL = (int(fpucurr.readline()))
                                CurrPU = (int(fpucurr.readline()))  # / 1
                                MVx1 = (int(fpucurr.readline())) + sz1
                                MVy1 = (int(fpucurr.readline())) + sz1
                                MVx2 = (int(fpucurr.readline())) + sz1
                                MVy2 = (int(fpucurr.readline())) + sz1
                        else:
                            with open(pufilenameCurrInfo) as fpucurr:
                                # CurrSL = (int(fpucurr.readline()))
                                CurrSL = 0
                                CurrPU = (int(fpucurr.readline()))  # / 1
                                MVx1 = (int(fpucurr.readline())) + sz1
                                MVy1 = (int(fpucurr.readline())) + sz1
                                MVx2 = (int(fpucurr.readline())) + sz1
                                MVy2 = (int(fpucurr.readline())) + sz1

                        SparseMethod = 0
                        testSR = (lfR == SparseND)
                        testNonR = (lfR == NonSparseND)
                        testSC = (lfC == SparseND)
                        testNonC = (lfC == NonSparseND)

                        # Defining Neighbours
                        N1PU = 0
                        N2PU = 0
                        N3PU = 0
                        N4PU = 0
                        N1SL = 0
                        N2SL = 0
                        N3SL = 0
                        N4SL = 0

                        if (METHODOLOGY):
                            Record_Depth1.append(
                                [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU,
                                 N3PU, N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu,
                                 pufilename4_ctu, viewNumber, PartNo])
                        else:
                            # print("Waqas Test")
                            # Getting only 5x5 key views
                            if (testSR.any() and testSC.any()):
                                # print("DCV",lfR,lfC)
                                t = 10
                            elif (testNonR.any() and testNonC.any()):
                                # print("case Sparse", lfR, lfC)
                                N_Ver_LC = lfC - 1
                                N_Ver_LR = lfR - 1
                                N_Ver_RC = lfC + 1
                                N_Ver_RR = lfR + 1

                                N_Ver_LC2 = lfC + 1
                                N_Ver_LR2 = lfR - 1

                                N_Ver_RC2 = lfC - 1
                                N_Ver_RR2 = lfR + 1

                                # print("KSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber), "NL (%d,%d)" % (N_Ver_LR, N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR, N_Ver_RC),"NL2 (%d,%d)" % (N_Ver_LR2, N_Ver_LC2),"NR2 (%d,%d)" % (N_Ver_RR2, N_Ver_RC2))

                                pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                if (os.path.exists(pufilename1)):
                                    with open(pufilename1) as fpu1:
                                        N1SL = (int(fpu1.readline()))
                                        N1PU = (int(fpu1.readline())) / 10
                                        # MVx1_N1 = (int(fpu1.readline()))
                                        # MVy1_N1 = (int(fpu1.readline()))
                                        # MVx2_N1 = (int(fpu1.readline()))
                                        # MVy2_N1 = (int(fpu1.readline()))

                                pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename2)
                                if (os.path.exists(pufilename2)):
                                    with open(pufilename2) as fpu2:
                                        N2SL = (int(fpu2.readline()))
                                        N2PU = (int(fpu2.readline())) / 10
                                        # MVx1_N2 = (int(fpu2.readline()))
                                        # MVy1_N2 = (int(fpu2.readline()))
                                        # MVx2_N2 = (int(fpu2.readline()))
                                        # MVy2_N2 = (int(fpu2.readline()))

                                pufilename3 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR2) + '_C' + str(
                                    N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename3_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR2) + '_C' + str(
                                    N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename3)
                                if (os.path.exists(pufilename3)):
                                    with open(pufilename3) as fpu3:
                                        N3SL = (int(fpu3.readline()))
                                        N3PU = (int(fpu3.readline())) / 10
                                        # MVx1_N3 = (int(fpu3.readline()))
                                        # MVy1_N3 = (int(fpu3.readline()))
                                        # MVx2_N3 = (int(fpu3.readline()))
                                        # MVy2_N3 = (int(fpu3.readline()))


                                pufilename4 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR2) + '_C' + str(
                                    N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename4_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR2) + '_C' + str(
                                    N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'


                                # print(pufilename4)
                                if (os.path.exists(pufilename4)):
                                    with open(pufilename4) as fpu4:
                                        N4SL = (int(fpu4.readline()))
                                        N4PU = (int(fpu4.readline())) / 10
                                        # MVx1_N4 = (int(fpu4.readline()))
                                        # MVy1_N4 = (int(fpu4.readline()))
                                        # MVx2_N4 = (int(fpu4.readline()))
                                        # MVy2_N4 = (int(fpu4.readline()))

                                # Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                #          N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu,
                                #          pufilename4_ctu,pufilenameCurrInfo_CTU])

                                # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 + ' N3: ' + pufilename3 + ' N4: ' + pufilename4)
                                # 04-15-23 N3SL written two times correcting it to N4SL
                                Record_Depth1.append(
                                    [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                     N4PU, N1SL, N2SL, N4SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber,PartNo])

                                # Record.append(
                                #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
                                #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,
                                #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
                                #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])

                            elif (testSR.any() and not testSC.any()):
                                N_Ver_LC = lfC - 1
                                N_Ver_LR = lfR
                                N_Ver_RC = lfC + 1
                                N_Ver_RR = lfR
                                pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename1)
                                if (os.path.exists(pufilename1)):
                                    with open(pufilename1) as fpu1:
                                        N1SL = (int(fpu1.readline()))
                                        N1PU = (int(fpu1.readline())) / 10
                                        MVx1_N1 = (int(fpu1.readline()))
                                        MVy1_N1 = (int(fpu1.readline()))
                                        MVx2_N1 = (int(fpu1.readline()))
                                        MVy2_N1 = (int(fpu1.readline()))

                                pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename2)
                                if (os.path.exists(pufilename2)):
                                    with open(pufilename2) as fpu2:
                                        N2SL = (int(fpu2.readline()))
                                        N2PU = (int(fpu2.readline())) / 10
                                        MVx1_N2 = (int(fpu2.readline()))
                                        MVy1_N2 = (int(fpu2.readline()))
                                        MVx2_N2 = (int(fpu2.readline()))
                                        MVy2_N2 = (int(fpu2.readline()))
                                N3PU = N1PU
                                N3SL = N1SL
                                N4PU = N2PU
                                N4SL = N2SL
                                # print("SSV [Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                                # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 )
                                # 04-15-23 N3SL written two times correcting it to N4SL
                                Record_Depth1.append(
                                    [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                     N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber,PartNo])

                            elif (testSC.any() and not testSR.any()):
                                N_Ver_LC = lfC
                                N_Ver_LR = lfR - 1
                                N_Ver_RC = lfC
                                N_Ver_RR = lfR + 1

                                pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                    N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename1)
                                if (os.path.exists(pufilename1)):
                                    with open(pufilename1) as fpu1:
                                        N1SL = (int(fpu1.readline()))
                                        N1PU = (int(fpu1.readline())) / 10
                                        MVx1_N1 = (int(fpu1.readline()))
                                        MVy1_N1 = (int(fpu1.readline()))
                                        MVx2_N1 = (int(fpu1.readline()))
                                        MVy2_N1 = (int(fpu1.readline()))

                                pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.txt'
                                pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                    N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '.png'

                                # print(pufilename2)
                                if (os.path.exists(pufilename2)):
                                    with open(pufilename2) as fpu2:
                                        N2SL = (int(fpu2.readline()))
                                        N2PU = (int(fpu2.readline())) / 10
                                        MVx1_N2 = (int(fpu2.readline()))
                                        MVy1_N2 = (int(fpu2.readline()))
                                        MVx2_N2 = (int(fpu2.readline()))
                                        MVy2_N2 = (int(fpu2.readline()))

                                N3PU = N1PU
                                N3SL = N1SL
                                N4PU = N2PU
                                N4SL = N2SL

                                # MVx1_N3 = MVx1_N1
                                # MVy1_N3 = MVy1_N1
                                # MVx2_N3 = MVx2_N1
                                # MVy2_N3 = MVy2_N1
                                #
                                # MVx1_N4 = MVx1_N2
                                # MVy1_N4 = MVy1_N2
                                # MVx2_N4 = MVx2_N2
                                # MVy2_N4 = MVy2_N2

                                # print("SSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                                # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2)
                                # 04-15-23 N3SL written two times correcting it to N4SL
                                Record_Depth1.append(
                                    [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                     N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,viewNumber,PartNo])

            # Record.append(
            #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
            #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename4_ctu, pufilename4_ctu,
            #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
            #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])


    if(len(Record_Depth1) == 0):
        print("No sample is found for depth 1 prediction")
        df=[]
    else:
        df = pd.DataFrame(np.array(Record_Depth1), columns=['FileName', 'LFname', 'lfR', 'lfC', 'rate', 'CTUno', 'CurrPU', 'CurrSL', 'N1PU',
                               'N2PU', 'N3PU', 'N4PU', 'N1SL', 'N2SL', 'N3SL', 'N4SL', 'N1_CTU', 'N2_CTU', 'N3_CTU',
                               'N4_CTU', 'viewNumber','PartNo'])
    return df



## Depth 2 reading
def LoadFiftySixViewsDepth2TestImages(LFImages,ctuPath,metaPath,metaPath_Test,ctuPath_Test,rate,PRINT_STATUS,Depth1file,METHODOLOGY):

    Record_Depth2=[]
    sz1 = 256
    sz2 = sz1 * 2
    MVX_1 = np.zeros(sz2)
    MVX_2 = np.zeros(sz2)
    MVY_1 = np.zeros(sz2)
    MVY_2 = np.zeros(sz2)
    MVX_Arr = []

    NonSparseD = [2, 4, 6, 8]
    SparseD = [1, 3, 5, 7, 9]
    SparseND = np.array(SparseD)
    NonSparseND = np.array(NonSparseD)

    depthVal=2


    # #------------------------------- [Reading Depth1 Information] --------------------------------------
    # Depth1pathDrive = '../../output/' + outputDNNFolder + '/Depth1/QP_' + str(rate) + '_TestImages/'
    # Depth1FileName = "Test_%s_DL_Depth%d_Rate_%d_LF%d.txt" % (Test, 1, int(rate), int(TestImage))
    # Depth1file=Depth1pathDrive + Depth1FileName
    # #---------------------------------------------------------------------------------------------------


    with open(Depth1file) as fD0:
        lines = fD0.readlines()
        # check_counter=0
        # count_ones=0
        for line in lines:
            Tline=line.split()
            # check_counter = check_counter + 1
            # if check_counter == 14590:
            #     print('STOP')
            if(Tline[9]=='1'):
              # count_ones+=count_ones
              viewNumber=int(Tline[0])
              lfC = math.floor((viewNumber-1) / 9) + 1
              lfR = (viewNumber-1) % 9 + 1
              CTUno = int(Tline[1])
              #print(viewNumber,lfR,lfC)
                  #print("Split Yes ",viewNumber, lfR,lfC,CTUno)
              PartNo=int(Tline[3])
              for PartNo2 in range(0, 4):
                lfname = LFImages
                CurrPU = 0
                CurrSL = 0
                pufilenameCurrInfo = metaPath_Test + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                    lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                pufilenameCurrInfo_CTU = ctuPath_Test + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                    lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo)+ '_p' + str(PartNo2) + '.png'

                # GT Values loaded
                pufilenameCurrInfo_GT = metaPath + lfname + '/HCI_' + lfname + '_R' + str(lfR) + '_C' + str(
                    lfC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(
                    PartNo2) + '.txt'

                pufilename1_ctu = pufilenameCurrInfo_CTU
                pufilename2_ctu = pufilenameCurrInfo_CTU
                pufilename3_ctu = pufilenameCurrInfo_CTU
                pufilename4_ctu = pufilenameCurrInfo_CTU

                if (os.path.exists(pufilenameCurrInfo)):
                    if (os.path.exists(pufilenameCurrInfo_GT)):
                        with open(pufilenameCurrInfo_GT) as fpucurr:
                            CurrSL = (int(fpucurr.readline()))
                            CurrPU = (int(fpucurr.readline()))  # / 1
                            MVx1 = (int(fpucurr.readline())) + sz1
                            MVy1 = (int(fpucurr.readline())) + sz1
                            MVx2 = (int(fpucurr.readline())) + sz1
                            MVy2 = (int(fpucurr.readline())) + sz1
                    else:
                        with open(pufilenameCurrInfo) as fpucurr:
                            # CurrSL = (int(fpucurr.readline()))
                            CurrSL = 0
                            CurrPU = (int(fpucurr.readline()))  # / 1
                            MVx1 = (int(fpucurr.readline())) + sz1
                            MVy1 = (int(fpucurr.readline())) + sz1
                            MVx2 = (int(fpucurr.readline())) + sz1
                            MVy2 = (int(fpucurr.readline())) + sz1


                    SparseMethod = 0
                    testSR = (lfR == SparseND)
                    testNonR = (lfR == NonSparseND)
                    testSC = (lfC == SparseND)
                    testNonC = (lfC == NonSparseND)

                    # Defining Neighbours
                    N1PU = 0
                    N2PU = 0
                    N3PU = 0
                    N4PU = 0
                    N1SL = 0
                    N2SL = 0
                    N3SL = 0
                    N4SL = 0

                    if (METHODOLOGY):
                        Record_Depth2.append(
                            [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU,
                             N3PU,
                             N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu,
                             pufilename3_ctu, pufilename4_ctu, viewNumber, PartNo, PartNo2])
                    else:
                        # print("Waqas Test")
                        # Getting only 5x5 key views
                        if (testSR.any() and testSC.any()):
                            # print("DCV",lfR,lfC)
                            t = 10
                        elif (testNonR.any() and testNonC.any()):
                            # print("case Sparse", lfR, lfC)
                            N_Ver_LC = lfC - 1
                            N_Ver_LR = lfR - 1
                            N_Ver_RC = lfC + 1
                            N_Ver_RR = lfR + 1

                            N_Ver_LC2 = lfC + 1
                            N_Ver_LR2 = lfR - 1

                            N_Ver_RC2 = lfC - 1
                            N_Ver_RR2 = lfR + 1

                            # print("KSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber), "NL (%d,%d)" % (N_Ver_LR, N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR, N_Ver_RC),"NL2 (%d,%d)" % (N_Ver_LR2, N_Ver_LC2),"NR2 (%d,%d)" % (N_Ver_RR2, N_Ver_RC2))

                            pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p'+ str(PartNo2) + '.png'

                            if (os.path.exists(pufilename1)):
                                with open(pufilename1) as fpu1:
                                    N1SL = (int(fpu1.readline()))
                                    N1PU = (int(fpu1.readline())) / 10
                                    # MVx1_N1 = (int(fpu1.readline()))
                                    # MVy1_N1 = (int(fpu1.readline()))
                                    # MVx2_N1 = (int(fpu1.readline()))
                                    # MVy2_N1 = (int(fpu1.readline()))

                            pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p'+ str(PartNo2) + '.png'

                            # print(pufilename2)
                            if (os.path.exists(pufilename2)):
                                with open(pufilename2) as fpu2:
                                    N2SL = (int(fpu2.readline()))
                                    N2PU = (int(fpu2.readline())) / 10
                                    # MVx1_N2 = (int(fpu2.readline()))
                                    # MVy1_N2 = (int(fpu2.readline()))
                                    # MVx2_N2 = (int(fpu2.readline()))
                                    # MVy2_N2 = (int(fpu2.readline()))

                            pufilename3 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR2) + '_C' + str(
                                N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename3_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR2) + '_C' + str(
                                N_Ver_LC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p'+ str(PartNo2) + '.png'

                            # print(pufilename3)
                            if (os.path.exists(pufilename3)):
                                with open(pufilename3) as fpu3:
                                    N3SL = (int(fpu3.readline()))
                                    N3PU = (int(fpu3.readline())) / 10
                                    # MVx1_N3 = (int(fpu3.readline()))
                                    # MVy1_N3 = (int(fpu3.readline()))
                                    # MVx2_N3 = (int(fpu3.readline()))
                                    # MVy2_N3 = (int(fpu3.readline()))


                            pufilename4 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR2) + '_C' + str(
                                N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename4_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR2) + '_C' + str(
                                N_Ver_RC2) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p'+ str(PartNo2) + '.png'


                            # print(pufilename4)
                            if (os.path.exists(pufilename4)):
                                with open(pufilename4) as fpu4:
                                    N4SL = (int(fpu4.readline()))
                                    N4PU = (int(fpu4.readline())) / 10
                                    # MVx1_N4 = (int(fpu4.readline()))
                                    # MVy1_N4 = (int(fpu4.readline()))
                                    # MVx2_N4 = (int(fpu4.readline()))
                                    # MVy2_N4 = (int(fpu4.readline()))

                            # Record.append([pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                            #          N4PU, N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu,
                            #          pufilename4_ctu,pufilenameCurrInfo_CTU])

                            # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 + ' N3: ' + pufilename3 + ' N4: ' + pufilename4)

                            Record_Depth2.append(
                                [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                 N4PU, N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber,PartNo,PartNo2])

                            # Record.append(
                            #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
                            #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,
                            #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
                            #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])

                        elif (testSR.any() and not testSC.any()):
                            N_Ver_LC = lfC - 1
                            N_Ver_LR = lfR
                            N_Ver_RC = lfC + 1
                            N_Ver_RR = lfR
                            pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p'+ str(PartNo2) + '.png'

                            # print(pufilename1)
                            if (os.path.exists(pufilename1)):
                                with open(pufilename1) as fpu1:
                                    N1SL = (int(fpu1.readline()))
                                    N1PU = (int(fpu1.readline())) / 10
                                    MVx1_N1 = (int(fpu1.readline()))
                                    MVy1_N1 = (int(fpu1.readline()))
                                    MVx2_N1 = (int(fpu1.readline()))
                                    MVy2_N1 = (int(fpu1.readline()))

                            pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.png'

                            # print(pufilename2)
                            if (os.path.exists(pufilename2)):
                                with open(pufilename2) as fpu2:
                                    N2SL = (int(fpu2.readline()))
                                    N2PU = (int(fpu2.readline())) / 10
                                    MVx1_N2 = (int(fpu2.readline()))
                                    MVy1_N2 = (int(fpu2.readline()))
                                    MVx2_N2 = (int(fpu2.readline()))
                                    MVy2_N2 = (int(fpu2.readline()))
                            N3PU = N1PU
                            N3SL = N1SL
                            N4PU = N2PU
                            N4SL = N2SL
                            # print("SSV [Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                            # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2 )

                            Record_Depth2.append(
                                [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                 N4PU, N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu, viewNumber,PartNo,PartNo2])

                        elif (testSC.any() and not testSR.any()):
                            N_Ver_LC = lfC
                            N_Ver_LR = lfR - 1
                            N_Ver_RC = lfC
                            N_Ver_RR = lfR + 1

                            pufilename1 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename1_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_LR) + '_C' + str(
                                N_Ver_LC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.png'

                            # print(pufilename1)
                            if (os.path.exists(pufilename1)):
                                with open(pufilename1) as fpu1:
                                    N1SL = (int(fpu1.readline()))
                                    N1PU = (int(fpu1.readline())) / 10
                                    MVx1_N1 = (int(fpu1.readline()))
                                    MVy1_N1 = (int(fpu1.readline()))
                                    MVx2_N1 = (int(fpu1.readline()))
                                    MVy2_N1 = (int(fpu1.readline()))

                            pufilename2 = metaPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.txt'
                            pufilename2_ctu = ctuPath + lfname + '/HCI_' + lfname + '_R' + str(N_Ver_RR) + '_C' + str(
                                N_Ver_RC) + '_ctuno' + str(CTUno) + '_d' + str(depthVal) + '_p' + str(PartNo) + '_p' + str(PartNo2) + '.png'

                            # print(pufilename2)
                            if (os.path.exists(pufilename2)):
                                with open(pufilename2) as fpu2:
                                    N2SL = (int(fpu2.readline()))
                                    N2PU = (int(fpu2.readline())) / 10
                                    MVx1_N2 = (int(fpu2.readline()))
                                    MVy1_N2 = (int(fpu2.readline()))
                                    MVx2_N2 = (int(fpu2.readline()))
                                    MVy2_N2 = (int(fpu2.readline()))

                            N3PU = N1PU
                            N3SL = N1SL
                            N4PU = N2PU
                            N4SL = N2SL

                            # MVx1_N3 = MVx1_N1
                            # MVy1_N3 = MVy1_N1
                            # MVx2_N3 = MVx2_N1
                            # MVy2_N3 = MVy2_N1
                            #
                            # MVx1_N4 = MVx1_N2
                            # MVy1_N4 = MVy1_N2
                            # MVx2_N4 = MVx2_N2
                            # MVy2_N4 = MVy2_N2

                            # print("SSV[Row:%d ,Col: %d, Viewno: %d]" % (lfR, lfC,viewNumber),"NL (%d,%d)"% (N_Ver_LR,N_Ver_LC),"NR (%d,%d)" % (N_Ver_RR,N_Ver_RC))

                            # print('Viewno.: ' + str(viewNumber) + ' RC '+ str(lfR)+'-'+str(lfC)  + ' N1: ' + pufilename1 + ' N2: '+ pufilename2)

                            Record_Depth2.append(
                                [pufilenameCurrInfo_CTU, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU,
                                 N4PU, N1SL, N2SL, N3SL, N3SL, pufilename1_ctu, pufilename2_ctu, pufilename3_ctu, pufilename4_ctu,viewNumber,PartNo,PartNo2])

                        # Record.append(
                        #     [pufilenameCurrInfo, lfname, lfR, lfC, rate, CTUno, CurrPU, CurrSL, N1PU, N2PU, N3PU, N4PU,
                        #      N1SL, N2SL, N3SL, N4SL, pufilename1_ctu, pufilename2_ctu, pufilename4_ctu, pufilename4_ctu,
                        #      pufilenameCurrInfo_CTU, MVx1_N1, MVy1_N1, MVx2_N1, MVy2_N1, MVx1_N2, MVy1_N2, MVx2_N2,
                        #      MVy2_N2, MVx1_N3, MVy1_N3, MVx2_N3, MVy2_N3, MVx1_N4, MVy1_N4, MVx2_N4, MVy2_N4])

    if(len(Record_Depth2) == 0):
        print("No sample is found for depth 2 ")
        df=[]
    else:
        df = pd.DataFrame(np.array(Record_Depth2), columns=['FileName', 'LFname', 'lfR', 'lfC', 'rate', 'CTUno', 'CurrPU', 'CurrSL', 'N1PU',
                               'N2PU', 'N3PU', 'N4PU', 'N1SL', 'N2SL', 'N3SL', 'N4SL', 'N1_CTU', 'N2_CTU', 'N3_CTU',
                               'N4_CTU', 'viewNumber','PartNo','PartNo2'])
    return df