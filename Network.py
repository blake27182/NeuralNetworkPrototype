
from Node import Node
from math import exp, trunc
from statistics import mean
import pandas as pd
import random


class Network(object):
    """docstring for Network"""
    def __init__(self, lDims):
        super(Network, self).__init__()
        self.layers = []            # list of list of Nodes
        self.layerDims = lDims      # tuple
        self.fileName = ""
        self.data = {}
        self.outputNames = []

        for i, x in enumerate(self.layerDims):
            self.layers.append([])
            for _ in range(x):
                if i+1 < len(self.layerDims):
                    self.layers[i].append(Node(self.layerDims[i + 1]))
                else:
                    self.layers[i].append(Node(0))

    @staticmethod
    def Sig(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def TruncIt(x):
        x *= 10000
        x = trunc(x)
        x /= 10000
        return x

    def PrintNet(self):
        print()
        for l in self.layers:
            actString = ""
            for i, n in enumerate(l):
                if i+1 < len(l):
                    actString += f"{n.activation:}    "
                else:
                    actString += f"{n.activation}"
            print(f"{actString:^50}")
            print()
        keys = list(self.data.keys())
        if keys:
            labels = ""
            for key in keys:
                labels += f"{key}  "
            labels = labels[0:-2]
            print(f"{labels:^50}")
        print()

    def Run(self, source):
        for i, n in enumerate(self.layers[0]):
            n.activation = source[i]

        for i, l in enumerate(self.layers):     # for each layer
            if i == 0:                          # skip the first layer
                continue
            for j, n in enumerate(l):           # for each node in the current layer
                actAverage = 0
                for k, p in enumerate(self.layers[i-1]):            # for each node in the previous layer
                    actAverage += p.activation * p.weights[j]       # add the weight to the current node * act of prev
                actAverage /= k
                n.activation = self.Sig(actAverage)

                if n.activation < .0001:
                    n.activation = 0
                elif n.activation > .9999:
                    n.activation = 1
                elif n.activation == .5:
                    pass
                else:
                    n.activation = self.TruncIt(n.activation)

    def FileReader(self, name):
        if name[-4:] == ".csv":
            self.fileName = name
        else:
            raise Exception("Must provide a .csv file")

    def SetData(self, filename):
        if type(filename) is dict:
            self.data = filename
        else:
            self.FileReader(filename)
            dF = pd.read_csv(self.fileName)
            columns = list(dF)
            for dirtyName in columns:
                if dirtyName[dirtyName.find(' ')+1:] == '1':
                    cleanName = dirtyName[0:dirtyName.find(' ')]
                    self.data[cleanName] = [[] for _ in range(self.layerDims[0])]

            for key in self.data.keys():
                for i in range(self.layerDims[0]):
                    data = dF[f"{key} {i+1}"]
                    self.data[key][i] = list(data)

    def CalcNudge(self, activation, weight, desAct):
        x = weight
        y = activation
        if desAct > self.Sig(x * y):
            wNudge = (exp(-x * y) * y)/10 / (1 + exp(-x * y)) ** 2
            aNudge = (exp(-x * y) * x)/50 / (1 + exp(-x * y)) ** 2
        elif desAct < self.Sig(x * y):
            wNudge = -(exp(-x * y) * y)/10 / (1 + exp(-x * y)) ** 2
            aNudge = -(exp(-x * y) * x)/50 / (1 + exp(-x * y)) ** 2
        else:
            wNudge = 0
            aNudge = 0

        return aNudge, wNudge

    def SetRandomWeights(self):
        for l in self.layers:
            for n in l:
                for w in range(len(n.weights)):
                    n.weights[w] = random.randint(1,100) / 500

    def BackProp(self):
        for i in range(len(self.layers)-2, -1, -1):
            for n in self.layers[i]:

                aTerts = []        # temp nudge list for current node
                for j in range(len(self.layers[i+1])):
                    aTert, wSemi = self.CalcNudge(n.activation,
                                                  n.weights[j],
                                                  self.layers[i+1][j].desAct)
                    aTerts.append(aTert)
                    n.wSemis[j].append(wSemi)

                n.aSemis.append(mean(aTerts))
                n.desAct = n.activation + mean(aTerts)

                if n.desAct > 1:
                    n.desAct = 1
                elif n.desAct < 0:
                    n.desAct = 0

    def SetNudges(self):
        for i, l in enumerate(self.layers):
            if i+1 == len(self.layers):
                break
            for n in l:
                for j in range(len(n.weights)):
                    n.weights[j] += mean(n.wSemis[j])

    def ClearSemiInfo(self):
        for l in self.layers:
            for n in l:
                n.ClearNudges()

    def Train(self, numTimes):
        for _ in range(numTimes):       # for the number of times to train
            keys = list(self.data.keys())
            for i in range(len(self.data[keys[0]])):    # for the number of full sets available
                for j, key in enumerate(keys):      # for each source:target pair
                    source = []
                    for lst in self.data[key]:      # for every list in a source bin
                        source.append(lst[i])       # build a source set
                    source = tuple(source)
                    self.Run(source)
                    # set desired activations with current targets
                    for k, n in enumerate(self.layers[-1]):     # for every node in the last layer
                        if k == j:
                            n.desAct = 1
                        else:
                            n.desAct = 0
                    # back propagate (set semi-final nudges)
                    self.BackProp()
                # set the new weights based on the averaged nudges
                for l in self.layers:
                    for n in l:
                        for m, weight in enumerate(n.weights):
                            weight += mean(n.wSemis[m])
                            n.weights[m] = weight
                # clear the semi-final info from net and nodes
                self.ClearSemiInfo()
            continue