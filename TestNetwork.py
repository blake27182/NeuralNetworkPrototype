from Network import Network

testWork = Network((2, 2, 2))
# testWork.SetData({"None": [], "Left": [], "Right": [], "Both": []})

# testWork.layers[0][0].weights[0] = -100
# testWork.layers[0][0].weights[1] = 100
# testWork.layers[0][0].weights[2] = 0
# testWork.layers[0][0].weights[3] = 0
#
# testWork.layers[0][1].weights[0] = 0
# testWork.layers[0][1].weights[1] = 0
# testWork.layers[0][1].weights[2] = -100
# testWork.layers[0][1].weights[3] = 100
#
# testWork.layers[1][0].weights[0] = 100
# testWork.layers[1][0].weights[1] = -100
# testWork.layers[1][0].weights[2] = 100
# testWork.layers[1][0].weights[3] = -100
#
# testWork.layers[1][1].weights[0] = -100
# testWork.layers[1][1].weights[1] = 100
# testWork.layers[1][1].weights[2] = -100
# testWork.layers[1][1].weights[3] = 100
#
# testWork.layers[1][2].weights[0] = 100
# testWork.layers[1][2].weights[1] = 100
# testWork.layers[1][2].weights[2] = -100
# testWork.layers[1][2].weights[3] = -100
#
# testWork.layers[1][3].weights[0] = -100
# testWork.layers[1][3].weights[1] = -100
# testWork.layers[1][3].weights[2] = 100
# testWork.layers[1][3].weights[3] = 100

# testSource = (0, 1)
# testWork.Run(testSource)

testWork.SetData("Spread.csv")
testWork.SetRandomWeights()
testWork.Train(1000)

tS = (0, 1)
testWork.Run(tS)
testWork.PrintNet()
tS = (1, 0)
testWork.Run(tS)
testWork.PrintNet()

