# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 11:51:43 2019

@author: Sadman Sakib
"""
from scipy.spatial import distance
import numpy as np
import random
import pandas as pd
import globalVars_q1 as gv 
#*****Input file and K value take input*****
def takeInput():
    gv.dataset = pd.read_csv("q1_input.csv")
    gv.k= int(input("Please Enter K: "))
    gv.maxCols=gv.dataset.shape[1]
    gv.maxRows=gv.dataset.shape[0]
    gv.medoidsPoints=random.sample(range(0, gv.maxRows), (gv.k))
    gv.medoidsPointsEveryIterations.append(gv.medoidsPoints)
    gv.X=np.asarray(gv.dataset)    
#*****Calculate distance between vectors*****
def calculateDistanceList(OneVector, AllVector):
    return (distance.cdist(OneVector, AllVector, gv.distanceType))
#*****Calculate distance matrix*****    
def calculateDistanceMatrix(medoids):
    dist=frq=0
    distanceMatrix=gv.dataset.copy()
    for pointsIndex in medoids:
        dist=calculateDistanceList([gv.X[pointsIndex]], gv.X)
        dist=dist.T
        distanceMatrix['Cluster '+str(frq)+" Distance"]=dist
        frq+=1    
    return distanceMatrix
takeInput()    
distanceMatrix=calculateDistanceMatrix(gv.medoidsPoints)
#*****Find the cluster points*****
def assignPointsToCluster():
    df_clusterDistanceAll=distanceMatrix.iloc[:,(gv.maxCols):]
    clusterDistance=np.asarray(df_clusterDistanceAll)
    assignedClusterList=[]
    for index in range(gv.maxRows):
        assignedClusterIndex=np.argmin(clusterDistance[index])
        assignedClusterList.append(assignedClusterIndex)
        assignedClusterArray=np.asarray(assignedClusterList).T
    distanceMatrix['Cluster_ID']=assignedClusterArray    
    return distanceMatrix
#*****Store the results to CSV*****
def saveResult(result_df): 
    save_df=result_df.iloc[:,:(gv.maxCols)]
    save_df['Cluster_ID']=result_df['Cluster_ID']
    print("Saving "+ 'clusters_'+str(gv.k)+'.csv . . . .')
    save_df.to_csv('clusters_'+str(gv.k)+'.csv', index=False)   
distanceMatrix=assignPointsToCluster()
gv.distanceMatrixEveryIter.append(distanceMatrix)
def runKemdoids():
    #*****Divide dataset according to clusters*****
    global distanceMatrix
    while (gv.runFlag):
        clusterOfOneIter=[]
        for region, df_region in distanceMatrix.groupby('Cluster_ID'):
            clusterOfOneIter.append(df_region)       
        # *****Now, calculated each pairwise distance sum *****
        newCentroidList=[]
        for eachClusterDF in clusterOfOneIter:
            #*****Get only the points vector*****
            df_clusterPointsOnly=eachClusterDF.iloc[:,:(gv.maxCols)] 
            clusterPointsOnly=np.asarray(df_clusterPointsOnly)
            clusterPointsOnlyIndex=df_clusterPointsOnly.index.values
            distanceSum=[]
            for point in clusterPointsOnly:
                dis=calculateDistanceList([point],clusterPointsOnly)
                distanceSum.append(np.sum(dis))
            newCentroidIndex=np.argmin(distanceSum)
            newCentroidList.append(clusterPointsOnlyIndex[newCentroidIndex])
        gv.medoidsPointsEveryIterations.append(newCentroidList)
        distanceMatrix=calculateDistanceMatrix(newCentroidList)
        distanceMatrix=assignPointsToCluster()
        gv.distanceMatrixEveryIter.append(distanceMatrix)    
        if(gv.iterations==100 or (gv.medoidsPointsEveryIterations[len(gv.medoidsPointsEveryIterations)-2]==newCentroidList)):
            gv.runFlag=False
            saveResult(gv.distanceMatrixEveryIter[gv.iterations])
        gv.iterations+=1    
runKemdoids()
#**** SILHOUTTE WIDTH Calculation****
def findSilhoutteWidth():    
    #***Split the dataframe by clusters***
    global distanceMatrix
    for region, df_region in distanceMatrix.groupby('Cluster_ID'):
        gv.clusterIndividual.append(df_region)    
    for iCluster in range(len(gv.clusterIndividual)):
    #***    Calculating the value of a and b for each points in cluster i ***
        dfClusterEach=gv.clusterIndividual[iCluster] #i-th cluster
        npClusterEach=np.asarray(dfClusterEach.iloc[:,:gv.maxCols])
        clusterTotalInstances=npClusterEach.shape[0]    
        for eachPoint in npClusterEach:
            clusterAllPointsDistance=0
            clusterAllPointsDistance=(np.sum(calculateDistanceList([eachPoint], npClusterEach)))
            a_i=clusterAllPointsDistance/clusterTotalInstances
            neighbour_b_i=-1
            for jCluster in range(len(gv.clusterIndividual)):
                if(iCluster!=jCluster):
                    dfClusterEach=gv.clusterIndividual[jCluster] #j-th cluster
                    npOtherClusterEach=np.asarray(dfClusterEach.iloc[:,:gv.maxCols])
                    otherClusterTotalInstances=npOtherClusterEach.shape[0]
                    otherClusterAllPointsDistance=0 
                    otherClusterAllPointsDistance=(np.sum(calculateDistanceList([eachPoint], npOtherClusterEach)))
                    b_i=otherClusterAllPointsDistance/otherClusterTotalInstances
                    if(neighbour_b_i==-1):
                        neighbour_b_i=b_i
                    elif(neighbour_b_i>b_i):
                        neighbour_b_i=b_i
            s=(neighbour_b_i-a_i)/max(a_i,neighbour_b_i)
            gv.S.append(s)
    return gv.S #Return all S values
print("Avg Silhouette Width=", np.mean(findSilhoutteWidth()))
      

    

