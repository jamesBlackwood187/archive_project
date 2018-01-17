import pandas as pd 
import numpy as np 
import glob
import os

def ReadMatches(path):
	extension = 'csv'
	os.chdir(path)
	res = [i for i in glob.glob('*.{}'.format(extension))]
	atps = [file for file in res if file[:3] == 'atp']
	atps = sorted(atps)
	return atps

def ConcatFiles(fileList):
	for j,file in enumerate(fileList):
		if j ==0:
			fullDF = pd.read_csv(file, sep = ',', index_col = False)
		else:
			print(file)
			thisDF = pd.read_csv(file, sep = ',', index_col = False)
			fullDF = fullDF.append(thisDF)
	return fullDF




if __name__ == '__main__':
	filePath = '/home/jbl/Documents/Programming/Sports Betting/tennis'	
	files = ReadMatches(filePath)
	fullDF = ConcatFiles(files)
	a = fullDF.to_csv('fullMatches.csv')
