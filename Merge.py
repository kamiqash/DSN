import numpy as np
import os.path
from os import path
from os.path import exists
from merge_depths_files import depth_merge


outputDNNFolder= 'FullLF_RateByRate'

OutputFolderMergedDepths = 'MergedDepths_FullLF_RateByRate'

InputFolderMergedViews = 'MergedViews_FullLF_RateByRate'

basepath = '../../output/' + outputDNNFolder + '/' + InputFolderMergedViews + '/'
output_path = '../../output/' + outputDNNFolder + '/' + OutputFolderMergedDepths + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

test = ["A","B","C"]

rate = [22,27,32]
#test = ["B"]
#rate = [22]
for tst in test:
  if tst == "A":
    LF_image = [14,23,28]
  if tst == "B":
    LF_image = [4,10,13]
    #LF_image = [10]
  if tst == "C":
    LF_image = [5,7,16]
  for x in LF_image:
    for r in rate:
      #Reading Depth0 file
      image = "DL_HCI_" + str(x) + "_Rate_" + str(r) + "_Test_" + tst +"_Depth0.txt"
      print(image)
      LoadedFile = np.loadtxt(open(basepath+image, "rb"), delimiter=" ", skiprows=0)
      row = len(LoadedFile)
      col = len(LoadedFile[0])
      print("Rows: ",row, "Col: ",col)

      dep0 = LoadedFile
      #print(dep0)

      #Reading Depth1 file
      image = "DL_HCI_" + str(x) + "_Rate_" + str(r) + "_Test_" + tst +"_Depth1.txt"
      print(image)
      #if path.exists(basepath+image):
      LoadedFile = np.loadtxt(open(basepath+image, "rb"), delimiter=" ", skiprows=0)
      row = len(LoadedFile)
      col = len(LoadedFile[0])
      print("Rows: ",row, "Col: ",col)

      dep1 = LoadedFile
      #np.savetxt(output_path+"dep0.txt", dep0,fmt="%s")
      #Reading Depth2 file
      image = "DL_HCI_" + str(x) + "_Rate_" + str(r) + "_Test_" + tst +"_Depth2.txt"
      print(image)
      #if path.exists(basepath+image):
      LoadedFile = np.loadtxt(open(basepath+image, "rb"), delimiter=" ", skiprows=0)
      row = len(LoadedFile)
      col = len(LoadedFile[0])
      print("Rows: ",row, "Col: ",col)

      dep2 = LoadedFile

      To_write = depth_merge(dep0,dep1,dep2)
      output_name = "O_HCI_" + str(x) +"_" + str(r) + "_vieworder.txt"
      #print(output_name)
      #print(output_path)
      #print(To_write)
      np.savetxt(output_path+output_name, To_write,fmt="%s")