from os import path
import numpy as np
import pandas as pd
import glob

def findgender0(path):
    all_data = glob.glob(path+"/*.psv")
    Gender0 =pd.DataFrame()
    for data in all_data:
        newdata= pd.read_csv(data,sep='|')
        
        if newdata['Age'][1] <40 and newdata['Age'][1] > 20:
              Gender0 = Gender0.append(newdata, ignore_index=True)
        
    return Gender0

path ="/Users/smile/Desktop/1.Semster/Data challege/training"
Gender0 =findgender0(path)
Gender0.to_csv('20<Age<40.csv',index=False)