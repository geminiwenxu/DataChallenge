import numpy as np
from numpy import nan as NA
from sklearn.impute import SimpleImputer
from os import path
import pandas as pd
import glob

imp = SimpleImputer(missing_values=NA, strategy = "mean")
def findgender0(path):
    all_data = glob.glob(path+"/*.psv")
    Gender0 =pd.DataFrame()
    cont = 0
    for data in all_data:
       newdata= pd.read_csv(data,sep='|') 
      
       if newdata['Age'][1] <20:#0 and newdata['Age'][1] >=40:
        featuredata =newdata.iloc[:,0:7]
        #newdf =pd.DataFrame(imp.fit_transform(featuredata))
        newdf = featuredata.interpolate(method='linear')
        
        newnwew =pd.concat([newdf,newdata.iloc[:,7:]],axis=1)
        Gender0 = Gender0.append(newnwew, ignore_index=True)
        cont = cont +1
    print(cont)
    return Gender0



path ="/Users/smile/Desktop/1.Semster/Data challege/training"
modelcol= pd.read_csv('p000001.psv',sep='|') 
Gender0 =findgender0(path)
#Gender0.columns = modelcol.columns
Gender0.to_csv('20<linear.csv',index=False)