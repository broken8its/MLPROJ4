from LogReg import LogReg
import numpy as np 



lr = LogReg
Data = lr.getData(lr, 'mushrooms.csv')
permutation = np.random.permutation(Data.shape[1])
trainData, testData = Data[permutation][6500:], Data[permutation][:6500]
predictions = lr.logReg(lr, trainData, 0.01)
