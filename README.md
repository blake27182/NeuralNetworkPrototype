# NeuralNetworkPrototype
This project is still being developed and only works about 80 percent of the time

This package is a bare-bones, but complete Single-Output Neural Network. 
It is fully modular and can be customized and trained for any application! 

I decided to do this project as a second-year in college as something to challenge my creativity. I knew that a neural network was a weighted graph of sorts, that it had to be trained, that it can take many inputs to resolve to a few outputs, and I knew that there was something called back-propagation that can inform the minor changes made to the network little by little. I isolated myself from doing any research further because I knew i had the necessary information to do it on my own. 

The first issue I ran into was that after being trained, the network was only capable of producing one output. Most of the time, this ouput was not based on the training, but on the random startup for the weights of the graph. After a week of turning it over in the back of my mind, I realized I needed to train ALL outputs before back-propogating. Then the network was more correct, but it was only capable of one output still.

After 3 solid days of toiling over how to adjust weights vs activation bias during back-propogation, I came up with a method of calculating the changes by way of taking the derivative of a surface which represents the cost of a node in x-y-z space where x is weight, y is activation, and z is the resulting cost. After applying this formula, the network worked much better, and became capable of multiple (correct) outputs after one training session.

As of now, the random startup sometimes influences the accuracy of the network some of the time. About 2 out of 10 training sessions result in a network that doesnt work.

INSTRUCTIONS FOR USE

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
