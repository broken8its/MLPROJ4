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
        mushShape = np.array(self.rawMushData.shape)
        mushShape[1] = mushShape[1] - np.intc(1)
        self.mushData = np.zeros(mushShape, dtype = np.intc)

        # Converts discreet letter values to ints
        for self.itt in range (mushShape[0]):
            mushDataTemp = np.zeros(mushShape[1], dtype = np.intc)
            dimVals = np.zeros(26, dtype = np.unicode_)
            valCounter = np.intc(0)
            for self.itt2 in range (mushShape[1]):
                for self.itt3 in range (26):
                    if self.rawMushData[self.itt, self.itt2 + np.intc(1)] == dimVals[self.itt3]:
                        mushDataTemp[self.itt2] = self.itt3
                        break
                    elif dimVals[self.itt3] == np.zeros(1, dtype = np.unicode_):
                        mushDataTemp[self.itt2] = valCounter
                        valCounter = valCounter + np.intc(1)
                        dimVals[self.itt3] = self.rawMushData[self.itt, self.itt2 + np.intc(1)]
                        break
            self.mushData[self.itt] = mushDataTemp
        return self.mushData

    #this function is doing the regression, should only take dataset and hyperParameters
    def logReg(self, dataset, learningRate):
        shapeOData = np.array(dataset.shape)
        self.Ws = np.zeros(shapeOData[0], dtype=np.double).T
        bias = np.zeros(shapeOData[1], dtype=np.double)
        self.Xs = dataset.T
        self.Ts = dataset[0].T
        self.yPred = np.zeros(self.Ts.shape,dtype=np.intc)
        m = len(self.Ts)
        for self.itt in range(np.shape(self.Xs)[0]):
            self.Xs[self.itt][0] = 1
        for self.itt in range(3000):
            Z = np.dot(self.Xs,self.Ws) + bias
            self.A = self.sigmoid(self, Z)
            loss = self.logLoss(self, self.A,self.Ts)
            dz = self.A - self.Ts
            dw = np.double(1/m) * np.dot(self.Xs.T,dz)
            dbias = np.sum(dz)
            self.Ws = self.Ws - learningRate * dw
            bias = bias - learningRate * dbias

            if(self.itt%100==0):
                print(loss)
        i = np.double(0)
        self.itt = np.intc(0)
        for i in range(self.A):
            itt=itt+1
            if i > np.double(.5):
                self.yPred[itt] = np.intc(1)
        


    def sigmoid(self, Z):
        return np.double(1) / (np.double(1) + np.e ** (-Z))

    def logLoss(self, yPred, target):
        return -np.mean(target * np.log(target) + (np.double(1) - target) * np.log(np.double(1) - yPred), dtype=np.double)
