import LogReg
import numpy as np 
import matplotlib.pyplot as plt

class main():
    Data = LogReg.getData('mushrooms.csv')
    predictions = LogReg.logReg(Data, 0.01)