import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class stockMarket:
    def __init__(self):
        '''
            Initialize the dataset
        '''
        self.loadData()
        
    def loadData(self):
        '''
            Loads the data from an hdf5 file
        '''
        with pd.HDFStore("../input/train.h5", "r") as train:
        # Note that the "train" dataframe is the only dataframe in the file
            self.train = train.get("train")
    
    def createEnvironment(self):
        '''
            Setting up the kagglegym environment
        '''
        # Create environment
        self.env = kagglegym.make()
        
        self.observation = self.env.reset()
        

    
    def viewData(self):
        '''
            Viewing the test dataframe for Kaggle
        '''
        '''
            print(self.observation.train.shape) 
            print(self.observation.features.head())
            print(self.observation.features.shape)
            print(self.observation.target.head())
            print(self.observation.target.shape)
            print(self.env.step(self.observation.target))
        '''
        
        '''
            looping through all observations
        '''
        while True:
        
            print(self.train.shape) 
            print(self.observation.features.head())
            print(self.observation.features.shape)
            print(self.observation.target.head())
            print(self.observation.target.shape)
            self.observation, reward, done, info = self.env.step(self.observation.target)
            
            if done:
                break;
        print(info)
        
    def handleNas(self):
        '''
            Handles the missing observations for columns 
        '''
        self.train = self.train.fillna(self.train.median())
    
    def fitOls(self):
        '''
            Fits a simple linear regression
        '''
        
        '''
            Splitting the data into train and test
        '''
        trainX, testX, trainY, testY = train_test_split( self.train.loc[:,'timestamp':'technical_44'], 
            self.train.loc[:,'y'], test_size=0.4, random_state=0)

        '''
            initializing and fitting a simple linear regression model
        '''
        reg = linear_model.LinearRegression()
        
        reg.fit(trainX, trainY)
        
        '''
            Adding a cross validation
        '''
        scores = cross_val_score(reg, trainX, trainY, scoring='r2') 
    
        
        print(scores)
        print(reg.coef_)
        
        
        print(reg.predict(self.observation.features.loc['timestamp':'technical_44', :]))
        
        
            
            
if __name__ == '__main__':
	stockObj = stockMarket()
	stockObj.createEnvironment()
	#stockObj.viewData()
	stockObj.handleNas()
	stockObj.fitOls()
