#!/usr/bin/python3
import pandas as pd


class stockMarket:
	def __init__(self):
		self.loadData()

	def loadData(self):
		with pd.HDFStore("../input/train.h5", "r") as allData:
			self.train = allData.get("train")
		print(self.train.head())

if __name__ == '__main__':
	stockObj = stockMarket()	
