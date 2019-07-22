class Node(object):
    """docstring for Node"""

    def __init__(self, n_size):
        super(Node, self).__init__()
        self.nextLayerSize = n_size
        self.weights = []
        self.wSemis = []          # list of lists of semi-final nudges
        self.activation = 0
        self.aSemis = []
        self.desAct = 0

        for _ in range(self.nextLayerSize):
            self.weights.append(0)
            self.wSemis.append([])

    def ClearNudges(self):
        self.desAct = 0
        self.aSemis = []
        for i in range(len(self.wSemis)):
            self.wSemis[i] = []