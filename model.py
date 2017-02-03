import kagglegym
import numpy as np
import pandas as pd



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
        
        print(self.observation.train.head())
        print(self.observation.train.shape) 
    
    def viewTest(self):
        '''
            Viewing the test dataframe for Kaggle
        '''
        print(self.observation.target.head())
        print(self.observation.target.shape)
        #print(self.env.step(self.observation.target))
        
        
        '''
            looping through all observations
        '''
        while True:
            observation, reward, done, info = self.env.step(self.observation.target)
            if done:
                break;
        print(info)
            
            
if __name__ == '__main__':
	stockObj = stockMarket()
	stockObj.createEnvironment()
	stockObj.viewTest()
