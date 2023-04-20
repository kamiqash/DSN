import numpy as np
import os.path
from os import path
from os.path import exists


def depth_merge(dep0,dep1,dep2):
  dep1_iter = 0;
  dep2_iter = 0;
  To_write = [];
  for ctu in range(len(dep0)):  #(len(dep0)):
      #print(ctu)
      #if no split write statement
      #else go in loop
      if dep0[ctu,-1]==0:
          #print(ctu)
          new=f'9 {int(dep0[ctu,1])} {int(dep0[ctu,2])} 9 9 9 9 9 9 9 9 9 9'
          To_write.append(new)
          #print(ctu,' writing statement')
      else:
          #print(ctu)
          #print(ctu,' going in loop')
          for cu_d1 in range(4):
              #Creating iterator for dep1
              if dep1[dep1_iter,-1]==0:
                  #print(dep1_iter,'writing depth 1 statment')
                  new=f'9 {int(dep1[dep1_iter,1])} {int(dep1[dep1_iter,2])} 9 9 9 9 9 9 9 9 9 9'
                  To_write.append(new)
                  dep1_iter=dep1_iter+1
              else:
                  #print(dep1_iter,'going to dep2 loop')
                  dep1_iter=dep1_iter+1
                  for cu_d2 in range(4):
                      #creating iterator for dep2
                      if dep2[dep2_iter,-1]==0:
                          new=f'9 {int(dep2[dep2_iter,1])} {int(dep2[dep2_iter,2])} 9 9 9 9 9 9 9 9 9 9'
                          To_write.append(new)
                          #print(dep2_iter,'writing depth 2 statment')
                          dep2_iter=dep2_iter+1
                      else:
                          new=f'9 {int(dep2[dep2_iter,1])} {int(dep2[dep2_iter,2]+1)} 9 9 9 9 9 9 9 9 9 9'
                          To_write.append(new)
                          new=f'9 {int(dep2[dep2_iter,1])} {int(dep2[dep2_iter,2]+1)} 9 9 9 9 9 9 9 9 9 9'
                          To_write.append(new)
                          new=f'9 {int(dep2[dep2_iter,1])} {int(dep2[dep2_iter,2]+1)} 9 9 9 9 9 9 9 9 9 9'
                          To_write.append(new)
                          new=f'9 {int(dep2[dep2_iter,1])} {int(dep2[dep2_iter,2]+1)} 9 9 9 9 9 9 9 9 9 9'
                          To_write.append(new)
                          #print(dep2_iter,'writing four statements')
                          dep2_iter=dep2_iter+1
      #print stop here
      if dep0[ctu,1]==63:
          #print('stop')
          new='stop'
          To_write.append(new)
  print("Depth 1: ",dep1_iter)
  print("Depth 2: ",dep2_iter)
  return To_write