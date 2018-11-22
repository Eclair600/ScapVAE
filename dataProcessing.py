#This file contains all the data preprocessing method
#that we are going to need we want all the meshes represented
#as numpy array of size 30.000+nb Measurements (3*10.000) with 10.000 being the number of points

# Some imports we are going to need

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
#import statsmodels.api as sm
import os
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import rcParams
import sklearn

filePathname = 'flattenMeshes_1.csv'

def preprocess(filePath):
    df_coordinates = pd.read_csv(filePath, encoding='ISO-8859-1' )
    #We delete the MeshID feature from our dataset
    df_coordinates.drop(df_coordinates.index[0], inplace=True)
    newDFnp = df_coordinates.values
    numberMeshes = len(newDFnp)
    allMeshCoordinates = np.zeros((numberMeshes,3,10000))

    i=0
    for mesh in newDFnp:
        meshCoordinates = np.zeros((3,10000))
        XcoordNp = np.zeros(10000)
        YcoordNp = np.zeros(10000)
        ZcoordNp = np.zeros(10000)
        for index in range(len(mesh)):
            if (index%3==0):
                XcoordNp[(index//3)-1] = mesh[index]
            if (index%3==1):
                YcoordNp[(index//3)-1] = mesh[index]
            if (index % 3 == 2):
                ZcoordNp[(index // 3)-1] = mesh[index]
        meshCoordinates[0] = XcoordNp
        meshCoordinates[1] = YcoordNp
        meshCoordinates[2] = ZcoordNp
        allMeshCoordinates[i] = meshCoordinates
        i+=1
    return allMeshCoordinates

print(len(preprocess(filePathname)))




