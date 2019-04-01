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
        self.mushData = np.zeros(mushShape, dtype = np.double)

        # Converts discreet letter values to ints
        for self.itt in range (mushShape[0]):
            mushDataTemp = np.zeros(mushShape[1], dtype = np.double)
            dimVals = np.zeros(26, dtype = np.unicode_)
            valCounter = np.double(0)
            for self.itt2 in range (mushShape[1]):
                for self.itt3 in range (26):
                    if self.rawMushData[self.itt, self.itt2 + np.intc(1)] == dimVals[self.itt3]:
                        mushDataTemp[self.itt2] = np.double(self.itt3)
                        break
                    elif dimVals[self.itt3] == np.zeros(1, dtype = np.unicode_):
                        mushDataTemp[self.itt2] = np.double(valCounter)
                        valCounter = valCounter + np.double(1)
                        dimVals[self.itt3] = self.rawMushData[self.itt, self.itt2 + np.intc(1)]
                        break
            self.mushData[self.itt] = mushDataTemp
        return self.mushData

    #this function is doing the regression, should only take dataset and hyperParameters
    def logReg(self, dataset, learningRate, testData):
        shapeOData = np.array(dataset.shape)
        Ws = np.zeros((shapeOData[1]-1,1), dtype=np.double)
        bias = np.zeros((1,1), dtype=np.double)
        Xs = dataset[:,1:]
        #Ts = np.zeros((shapeOData[1],1), dtype=np.intc)
        Ts = np.array(dataset[:,0])
        Ts = Ts.reshape(shapeOData[0], 1)
        self.yPred = np.zeros((shapeOData[0],1),dtype=np.intc)
        m = len(Ts)
        #for self.itt in range(np.shape(Xs)[0]):
            #Xs[self.itt][0] = 1
        Z = np.zeros((shapeOData[0],1), dtype = np.double)
        for self.itt in range(10000):
            Z = np.dot(Xs,Ws) + bias
            A = self.sigmoid(self, Z)
            loss = self.logLoss(self, A,Ts)
            dz = A - Ts
            dw = np.double(1/m) * np.dot(Xs.T,dz)
            dbias = np.sum(dz)
            Ws = np.add(Ws, -learningRate * dw)
            bias = bias - learningRate * dbias

            if(self.itt%100==0):
                print(loss)
        i = np.double(0)
        self.itt = np.intc(0)
        for i in (self.sigmoid(self, Z) ):
            if i > np.double(.5):
                self.yPred[self.itt] = np.intc(1)
            self.itt=self.itt + 1
        
        testZ = np.zeros((testData.shape[0], 1), dtype = np.double)
        testPreds = np.zeros((testData.shape[0], 1), dtype = np.double)
        testXs = testData[:,1:]
        testTs = testData[:,0]
        testTs = testTs.reshape(testData.shape[0], 1)
        testZ = np.dot(testXs, Ws) + bias
        i = np.double(0)
        self.itt = np.intc(0)
        for i in self.sigmoid(self, testZ):
            if i > .5:
                testPreds[self.itt] = 1
            self.itt = self.itt + 1
        errCount = np.intc(0)
        for self.itt in range (testTs.shape[0]):
            if testTs[self.itt] != testPreds[self.itt]:
                errCount = errCount + np.intc(1)
        testErr = np.double(np.double(errCount) / (self.itt + 1))
        print(testErr)
        



    def sigmoid(self, Z):
        return np.double(1) / (np.double(1) + np.exp(-Z))

    def logLoss(self, yPred, target):
        return -np.mean(target * np.log(yPred) + (np.double(1) - target) * np.log(np.double(1) - yPred), dtype=np.double)

    
