# NeuralNetworkPrototype
This project is still being developed and only works about 80 percent of the time

This package is a bare-bones, but complete Single-Output Neural Network. 
It is fully modular and can be customized and trained for any application! 

Step 1:
Define the dimensions of your network by the number of nodes you want in each layer

Step 2:
Either set the weights manually in your script (which would take forever if you have a large one) 
or load a formatted csv file with the source and target data. There is an example included called "spread.csv"
Since this is a Single-Output Network, the header of each column should be labeled 
as such "{target label} {index of input node}" and the data in the column should be the list of inputs for that input node. 

Step 3:
Train the network by using the Train() method and passing the number of training cycles you wish to perform

Step 4:
Use it! by using the Run() method, you can pass inputs to the network and it will run.

Step 5:
See your output! Currently, you can access your output in two ways: accessing the last index of the Network's
layers data member, or by using the PrintNet() method, which will print the activations and labels to the console.

Enjoy your new, one-of-a-kind Einstein3000!
