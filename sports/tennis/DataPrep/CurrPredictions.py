import pandas as pd
import numpy as np
import FormTrainTest


file = '/home/jbl/Documents/Programming/Sports Betting/tennis/Data/frenchOpen1.csv'
fullDFs = pd.read_csv(file, index_col = 'ix')

fullDFs['StringScore'] = fullDFs['score'].astype(str)
fullDFs['NumSets']     = fullDFs['StringScore'].apply(lambda x: x.count('-'))


FeatureSet = FormTrainTest.FormDataSet(fullDFs)


a = FeatureSet.to_csv('/home/jbl/Documents/Programming/Sports Betting/tennis/Data/frenchOpen.csv')


