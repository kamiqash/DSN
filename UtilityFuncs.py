
import pandas as pd

def Stats_Split_Non_Split(Input_df,DataInfo):
    Split = Input_df[Input_df['CurrSL'] == '1']
    Non_Split = Input_df[Input_df['CurrSL'] == '0']
    percentsplit = len(Split) / len(Input_df)
    percentnonsplit = len(Non_Split) / len(Input_df)

    print(DataInfo,"Samples",len(Input_df), "Split/NonSplit Count [%d / %d] " % (len(Split),len(Non_Split) ), "Split/NonSplit Perct [%.02f / %.02f] " % (percentsplit * 100,percentnonsplit * 100))

    # print("%s Total Samples: %d"% (DataInfo,len(Input_df)))
    # print("%s ",DataInfo ," splits    : ", len(Split), "Percent Split", percentsplit * 100)
    # print("%s ",DataInfo ," non splits: ", len(Non_Split), "Percent non Split", percentnonsplit * 100)