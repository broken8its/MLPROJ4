from LogReg import LogReg
import numpy as np 



lr = LogReg
Data = lr.getData(lr, 'mushrooms.csv')
predictions = lr.logReg(lr, Data, 0.01)
