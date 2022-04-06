import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

pd.set_option("display.precision", 2)
DATA_URL = "./"
np.random.seed(307)

#Take the sigmoid function of t
def sigmoid(t):
    return(1/(1+np.exp(-t)))

#get the negative log likelihood (cost function)
#we add a little noise to sigma to avoid a divide by zero error
def negLogLikehood(y_train, sigma):
    return(-np.sum(y_train * np.log(sigma + 0.00001) + (1-y_train) * np.log((1-sigma) + 0.00001)))

#normalise the numeric columns in the auto dataset
def normalise(X):
    columnsToNormalise = []
    for i in range(len(X.columns)):
        if(X.dtypes[i] == 'int64'):
            columnsToNormalise.append(X.columns[i])
    for i in columnsToNormalise:
        X[i] = (X[i] - X[i].mean())/X[i].std()
    return(X)

def loadAuto():
    df = pd.read_csv(DATA_URL + "auto.csv", na_values='?')

    #create a new column called High
    df['High'] = np.where(df['mpg'] > 22, 1, 0)

    #fill the N/A values of horsepower and convert it into int64
    df[['horsepower']] = df[['horsepower']].fillna(value=df['horsepower'].mean())
    df["horsepower"] = df["horsepower"].astype(np.int64)
    
    #perform one-hot encoding for year and origin to use them for prediction
    encoder = OrdinalEncoder()
    df['year'] = encoder.fit_transform(df[['year']])
    df['origin'] = encoder.fit_transform(df[['origin']])

    #divide the dataset into features and label
    X = normalise(df[['horsepower', 'weight', 'year', 'origin']])
    y = df['High']

    return [X, y]

#The logistic regression class provides methods for fit, predict, and score
class logisticRegression:
    def __init__(self) -> None:
        self.weights = []
        self.b0 = 0
        self.predictions = []

    def fit(self,X_train, y_train, learningRate = 0.001, epochs = 10000, weightRange = 0.7):
        #get initial randomised weights
        self.weights = np.random.uniform(low=-weightRange, high=weightRange, size=(len(X_train.columns),))

        cost_value = []
        for i in range(epochs):
            t = self.b0 + np.dot(self.weights, X_train.T)
            sigma = sigmoid(t)
            db= np.sum(sigma - y_train)
            dw = np.dot(sigma - y_train,X_train)
            self.weights = self.weights - learningRate * dw
            self.b0 = self.b0 - learningRate * db
            cost_value.append(negLogLikehood(y_train, sigma))

    #get predictions for some data, where 1 is assigned for probabilities > 0.5, 0 otherwise
    def predict(self, X):
        predictions = sigmoid(np.dot(X, self.weights.T))
        classifications = [1 if i>0.5 else 0 for i in predictions]
        self.predictions = classifications
        return classifications

    #find the error rate given a set of features and labels
    def errorRate(self, X, y):
        self.predict(X)
        return(np.sum(np.square(y-self.predictions))/len(y)) 

#this is where the control flow of our main program starts
if __name__=="__main__":

    #load the auto dataset and do necessary preprocessing
    [X, y] = loadAuto()

    #split into a test and a train dataset (half and half) using a random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=307)

    #initiate the logistic regression instance
    lrm = logisticRegression()

    #fit the model
    lrm.fit(X_train, y_train)

    #calculate the training error rate
    trainingErrorRate = lrm.errorRate(X_train, y_train)
    print("The training error rate on this dataset is: "+str(trainingErrorRate)+"\n")

    #calculate the test error rate of the model
    testErrorRate = lrm.errorRate(X_test, y_test)
    print("The test error rate on this dataset is: "+str(testErrorRate)+"\n")


