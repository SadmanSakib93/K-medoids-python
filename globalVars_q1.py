# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 11:58:12 2019

@author: Sadman Sakib
"""

import random
import pandas as pd
import numpy as np
#Declare all global variables
dataset = pd.DataFrame()
k= -1
maxRows=-1
maxCols=-1
medoidsPointsEveryIterations=[]
medoidsPoints=[]
distanceMatrixEveryIter=[]
clusterIndividual=[]
S=[]
runFlag=True
iterations=1
distanceType='minkowski'