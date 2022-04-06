import numpy as np
import pandas as pd
from itertools import product

#we set some print settings so that the output clusters look decent and fit on the screen
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

#we load the ncidata from the txt file
data = pd.read_csv("ncidata.txt", delim_whitespace=True, header = None)
#define the linkage here
linkage = "Average" #Single, Average, Centroid, or Complete

#this is the class for each node in the dendrogram
class node:
    def __init__(self, data, height, L, R) -> None:
        #this is for when we initially make every observation a cluster
        if height == 0:                             #if the height is 0,
            self.leaf = True                        #this is a leaf
            self.height = 0
            self.observations = np.array([data])    #and simply has one observation
        else:
            self.leaf = False                       #if the height is not 0, this is an internal node
            #and we use the generate method to make this node and set the height, left and right node, and observations for the new node
            self.height, self.leftNode, self.rightNode, self.observations = self.generate(height, L, R)
     
    #the generate method takes the previous height and returns the new height, left node, right node, and observations     
    def generate(self, height, L, R):
        distance = calcDist(L, R)
        return distance + height, L, R, np.concatenate((L.getObservations(), R.getObservations()))

    #this method simply returns the observations
    def getObservations(self):
        #if this node is a leaf, we simply return the current observations
        if self.height == 0:
            return np.array(self.observations)
        #otherise, we return the observations of the left and right children, recursing till we hit all the leaves
        else:
            leftChildren = self.leftNode.getObservations()
            rightChildren = self.rightNode.getObservations()
            return np.concatenate((leftChildren, rightChildren))

#the calcDist method relies on the linkage variable defined globally at the start of the file
#2(a) - here we defined the different distance methods
def calcDist(x, y):
    #we first get the observations for both clusters
    xObservations = x.getObservations()
    yObservations = y.getObservations()
    
    #then take the number of observations in each cluster
    x_rows, x_cols = xObservations.shape
    y_rows, y_cols = yObservations.shape
    
    #we check for what the linkage type we are using is
    if(linkage == "Centroid"):
        #in case of centroid, we find the centroids for both clusters by taking 
        #the mean of all the observations and return the norm of the centroids
        centroidX = np.mean(xObservations, axis=0)
        centroidY = np.mean(yObservations, axis=0)
        
        return(np.linalg.norm(np.subtract(centroidX, centroidY)))
    elif(linkage == "Average"):
        #in case of average, we find the distances between each pair of observations in the two clusters
        #and return the average of their norms
        distances = []
        for comb in product(xObservations, yObservations):
            distances.append(np.linalg.norm(np.subtract(comb[0], comb[1])))
        return np.mean(distances)
    elif(linkage == "Complete"):
        #in case of complete, we find and return the max distance between two observations in each of the clusters
        maxDist = 0
        for i in range(x_rows):
            for j in range(y_rows):
                firstCoord = xObservations[i,:]
                secondCoord = yObservations[j,:]
                distance = np.linalg.norm(np.subtract(firstCoord, secondCoord))
                if distance > maxDist:
                    maxDist = distance

        return maxDist
    else:
        #in case of single linkage (or by default), we return the minimum distance between two observations in each of the clusters
        minDist = np.inf
        for i in range(x_rows):
            for j in range(y_rows):
                firstCoord = xObservations[i,:]
                secondCoord = yObservations[j,:]
                distance = np.linalg.norm(np.subtract(firstCoord, secondCoord))
                if distance < minDist and distance > 0:
                    minDist = distance

        return minDist
  

#our getClusters method takes the number of clusters to be output, K, and the dendrogram itself, which is a dictionary storing the
#different clusters that we got while reducing the number of clusters from number of observations to 1
#in addition to this, it error checks for K and outputs an error if K is larger than the number of observations
#2(b)
def getClusters(K, dendrogram):
    try:
        clusters = dendrogram[K]
        
        for i in range(len(clusters)):
            observations = clusters[i].getObservations()
            print("The observations in cluster number "+str(i+1)+" are:\n", observations,"\n")
            print("This cluster has "+str(len(observations))+" observations\n")
            
        return [[cluster.getObservations()] for cluster in clusters]
    except KeyError:
        print("You've entered a K greater than the number of initial observations or lesser than 1.\n Please enter an appropriate value of K.")

ncidata = data.transpose().to_numpy()
#we initially define one cluster per observation
clusters = np.array([node(data.loc[i,:], 0, None, None) for i in range(len(ncidata[:,1]))])
#we define an empty dendrogram and add the clusters for number of observations (i.e., separate nodes for every observation)
dendrogram = { len(clusters): clusters }

#we iterate till we have only one cluster left
while(clusters.size != 1):
    #we define the minimum distance between two clusters initially as 0
    minDist = np.inf
    x = -1
    y = -1
    
    #we iterate through the clusters pairwise in a way that we don't repeat pairs
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            firstCluster = clusters[i]
            secondCluster = clusters[j]
            #we find their distance according to the linkage measure we are using
            distance = calcDist(firstCluster, secondCluster)
            
            #if this distance is lower than the minimum distance, we store this distance and the indices of the current observations
            if distance <= minDist:
                minDist = distance
                x = i
                y = j
    
    xCluster = clusters[x]
    yCluster = clusters[y]
    newObservations = np.concatenate((xCluster.getObservations(), yCluster.getObservations()))
    #we append the new cluster with multiple clusters inside it to the clusters array
    clusters = np.append(clusters, node(newObservations, minDist, xCluster, yCluster))
    
    #we reset the above defined variables we use to find the clusters
    minDist = np.inf

    #and remove the clusters that we found to be the closest from the array
    clusters = np.delete(clusters, [x, y])
    
    #finally, before going to the next iteration, we add this set of clusters to the dendrogram dict
    dendrogram[len(clusters)] = clusters

#here we print and get back the clusters from the getClusters method according to the K we give it
finalClusters = getClusters(32, dendrogram)