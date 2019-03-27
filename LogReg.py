import numpy as np

class LogReg:
    def __init__(self):
        self.itt = np.intc(0)

    def sigmoid(self, Z):
        return np.double(1) / (np.double(1) + np.e ** (-Z))

    def logLoss(self, yPred, target):
        return -np.mean(target * np.log(target) + (np.double(1) - target) * np.log(np.double(1) - yPred))
