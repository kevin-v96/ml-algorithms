from dataclasses import replace
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def avg(target, indx):
    if len(indx) == 0:	return 0.0
    return sum([ target[i] for i in indx ]) / len(indx)

def rss(target,indx):
    if len(indx) == 0:	return 0.0
    mean = avg(target,indx)
    return sum([ pow( target[i]-mean , 2.0 ) for i in indx ])

class decisionTree:
    def __init__(self,data,target,indx,depth):					
        if depth==0 or len(indx) == 0:								
            self.leaf = True						
            self.prediction = avg(target,indx)
        elif len(set(target.values)) == 1:
            self.leaf = True							
            self.prediction = target.iloc[0]	
        else:											
            self.leaf = False			
            self.attr , self.split , self.L , self.R = self.generate(data,target,indx,depth)

    def generate(self,data,target,indx,depth):				
        p = len(data.columns)-1
        labels = [target[i] for i in indx]
        #1(b) - we only use p/3 features for training
        choices = np.random.choice(data.columns, int(np.ceil(p/3)), replace = False)
        opt = pow ( max(labels) - min(labels) , 2.0 ) * len(indx) + 1.0
        for j in choices:								
            all_cuts = set( [data[j][i] for i in indx] )
            for cut in all_cuts:
                yl = [ i for i in indx if data[j][i]<=cut ]
                yr = [ i for i in indx if data[j][i]>cut ]
                tmp = rss(target,yl) + rss(target,yr)
                if tmp < opt:
                    opt , attr , split, L , R = tmp , j , cut , yl , yr
        return attr , split , decisionTree(data,target,L,depth-1) , decisionTree(data,target,R,depth-1)

    def predict(self,x):
        if self.leaf == True:    return self.prediction
        if (x.loc[self.attr] <= self.split):    return self.L.predict(x)
        return self.R.predict(x)

#most of the program above this, other than a few indexing changes, is the tree as defined in boost.py from the course moodle
#we firstly load the boston dataset, add column names to it, add the medv column to it
Boston = load_boston()
bostondf = pd.DataFrame(Boston.data, columns = Boston.feature_names)
bostondf["MEDV"] = Boston.target
bostondf.columns = bostondf.columns.str.lower()

#here we split it into train and test sets
X_train, X_test, y_train, y_test = train_test_split(bostondf.loc[:, "crim":"lstat"], bostondf.loc[:, "medv"], train_size=0.5, random_state=307)

#we define the B and empty arrays that we will use later
B = 1
random_forest = []
tree_indices = []
n  = len(X_train)

#1(a) - this is where we generate B bootstrapped training sets and train them
for j in range(B):
    bts = np.unique(np.random.choice(y_train.index, n))
    tree_indices.append(bts)
    #1(b) - we train a tree with this bts, of height h (3 in case of the first part)
    tree = decisionTree(X_train, y_train, bts , 3)
    random_forest.append(tree)

#we use the B trees trained above to find the training and test OOBs
training_OOB = []
for idx in X_train.index:
    diff = []
    for tree_idx in range(len(random_forest)):
        if(idx not in tree_indices[tree_idx]):
            sample = X_train.loc[idx]
            tree = random_forest[tree_idx]
            trueLabel = y_train[idx]
            prediction = tree.predict(sample)
            diff.append(trueLabel - prediction)
    training_OOB.append(np.sum(np.square(diff)))

test_OOB = []
for idx in X_test.index:
    diff = []
    for tree_idx in range(len(random_forest)):
        if(idx not in tree_indices[tree_idx]):
            sample = X_test.loc[idx]
            tree = random_forest[tree_idx]
            trueLabel = y_test[idx]
            prediction = tree.predict(sample)
            diff.append(trueLabel - prediction)
    test_OOB.append(np.sum(np.square(diff)))

#1(c) - We calculate the training and test MSEs
training_MSE = np.mean(training_OOB) / len(X_train)
test_MSE = np.mean(test_OOB) / len(X_test)
print("Training MSE = ",training_MSE)
print("Test MSE = ", test_MSE)