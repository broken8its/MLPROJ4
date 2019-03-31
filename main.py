from LogReg import LogReg
import numpy as np 
import matplotlib.pyplot as plt


lr = LogReg
Data = lr.getData(lr, 'mushrooms.csv')
predictions = lr.logReg(lr, Data, 0.01)