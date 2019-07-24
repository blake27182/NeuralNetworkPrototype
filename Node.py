class Node(object):
    """Node is an element of Network"""

    def __init__(self, n_size):
        super(Node, self).__init__()
        self.nextLayerSize = n_size
        self.activation = 0
        self.weights = []
        self.aSemis = []        #
        self.wSemis = []        # list of lists of semi-final nudges
        self.desAct = 0

        for _ in range(self.nextLayerSize):
            self.weights.append(0)
            self.wSemis.append([])

    def ClearNudges(self):
        self.desAct = 0
        self.aSemis = []
        for i in range(len(self.wSemis)):
            self.wSemis[i] = []