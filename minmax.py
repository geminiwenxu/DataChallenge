from os import path
import pandas as pd
import glob

def findmax(path):
    all_data = glob.glob(path+"/*.psv")
    maxset =pd.DataFrame()
    for data in all_data:
        newdata= pd.read_csv(data,sep='|')
        newdata.min(axis = 0, skipna = True)
        
        maxset = maxset.append(newdata.max(axis = 0, skipna = True),ignore_index=True)
        
    return maxset
''''
def findmin(path):
    all_data = glob.glob(path+"/*.psv")
    maxset =pd.DataFrame()
    for data in all_data:
        newdata= pd.read_csv(data,sep='|')
        newdata.min(axis = 0, skipna = True)
        
        maxset = maxset.append(newdata.min(axis = 0, skipna = True),ignore_index=True)
        
    return maxset
''''
path ="/Users/smile/Desktop/1.Semster/Data challege/training"
maxset =findmax(path)
maxset.to_csv('max.csv',index=False)
