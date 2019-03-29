import numpy as np

__name__ = "LogReg"

class LogReg:
    def __init__(self):
        self.itt = np.intc(0)
        self.itt2 = np.intc(0)
        self.itt3 = np.intc(0)

    def getData(self, filename):
        self.rawMushData = np.loadtxt(filename, delimiter = ',', dtype = np.unicode_)
        self.dimLabels = self.rawMushData[0]
        self.rawMushData = self.rawMushData.T

        # Gets shape of data and subtracts labels row
        mushShape = self.rawMushData.shape
        mushShape[1] = mushShape[1] - np.intc(1)
        self.mushData = np.zeros(mushShape, dtype = np.intc)

        # Converts discreet letter values to ints
        for self.itt in range (mushShape[0]):
            mushDataTemp = np.zeros(mushShape[1], dtype = np.intc)
            dimVals = np.zeros(26, dtype = np.unicode_)
            valCounter = np.intc(0)
            for self.itt2 in range (mushShape[1]):
                for self.itt3 in range (26):
                    if self.rawMushData[self.itt, self.itt2] == dimVals[self.itt3]:
                        mushDataTemp[self.itt2] = self.itt3
                        break
                    elif dimVals[self.itt3] == np.unicode_(0):
                        mushDataTemp[self.itt2] = valCounter
                        valCounter = valCounter + np.intc(1)
                        dimVals[self.itt3] = self.rawMushData[self.itt, self.itt2]
                        break
            self.mushData[self.itt] = mushDataTemp
        return self.mushData

    #this function is doing the regression, should only take dataset and hyperParameters
    def logReg(self, dataset, learningRate):
        shapeOData = dataset.shape()
        Ws = np.zeros((shapeOData[1],shapeOData[0]), dtype=np.double)
        bias = np.zeros(shapeOData[0], dtype=np.double)
        Xs = dataset.T
        Ts = dataset[0].T
        yPred = np.zeros(Ts.shape(),dtype=np.intc)
        m = len(Ts)
        for self.itt in range(np.shape(Xs)[0]):
            Xs[self.itt][0] = 1
        for self.itt in range(3000):
            Z = np.dot(Xs,Ws) + bias
            self.A = self.sigmoid(Z)
            loss = self.logLoss(self.A,Ts)
            dz = self.A - Ts
            dw = np.double(1/m) * np.dot(Xs.T,dz)
            dbias = np.sum(dz)
            Ws = Ws - learningRate * dw
            bias = bias - learningRate * dbias

            if(self.itt%100==0):
                print(loss)
        i = np.double(0)
        self.itt = np.intc(0)
        for i in range(self.A):
            itt=itt+1
            if i > np.double(.5):
                yPred[itt] = np.intc(1)
        


    def sigmoid(self, Z):
        return np.double(1) / (np.double(1) + np.e ** (-Z))

    def logLoss(self, yPred, target):
        return -np.mean(target * np.log(target) + (np.double(1) - target) * np.log(np.double(1) - yPred), dtype=np.double)
