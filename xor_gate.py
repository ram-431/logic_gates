import numpy as np 

def initialize_weights(a,b):

    #initialize the required matrix with random gaussian values
    synaptic_weights=np.random.rand(a,b)
    return synaptic_weights


#defining sigmoid function

def sigmoid(x):
    return(1/(1+np.exp(-x)))


#defining derivative of the sigmoid function 
def sigmoid_der(x):
    return (x*(1-x))


#the overall activation of the perceptron
#takes the input from the previous node and generates the output from it using weights and the biases
def learn_perceptron(bias,weights,inputs):
    return sigmoid(np.dot(inputs,weights)+bias)


#function to calculate the value of the weights
#or the learning function for the neural networks 
def train_the_network(inputs,expected_outputs,weights,bias,learning_rate,training_iterations):
    for epoch in range(training_iterations):

        #forward pass in the network
        predicted_outputs=learn_perceptron(bias,weights,inputs)

        #backward pass in the network
        #calculation of the error
        error=sigmoid_der(predicted_outputs*(expected_outputs-predicted_outputs))


        #now calcualte the weights and the bias factor 
        weight_factor= np.dot(np.transpose(inputs),error)*learning_rate
        bias_factor= error*learning_rate

        #adjust the weights 
        weights +=weight_factor

        #adjust the bias
        bias +=bias_factor

        if ((epoch%1000)==0):
            print("Epoch ", epoch)
            print("predicted Output = ",np.transpose(predicted_outputs))
            print("Exprcted Output = ",np.transpose(expected_outputs))
            print()
        return weights

        





