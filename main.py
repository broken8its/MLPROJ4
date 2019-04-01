from LogReg import LogReg
import numpy as np 



lr = LogReg
Data = lr.getData(lr, 'mushrooms.csv')
Data = Data.T
permutation = np.random.permutation(Data.shape[0])
trainData, testData = Data[permutation][6500:], Data[permutation][:6500]
lr.logReg(lr, trainData, 0.01, testData)
