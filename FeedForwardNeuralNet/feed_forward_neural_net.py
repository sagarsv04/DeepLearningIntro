import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
# np.seterr(over='ignore')

class NeuralNetwork():
    def __init__(self):
        # seed the random number generator
        np.random.seed(1)
        # create dict to hold weights
        self.weights = {}
        # set initial number of layer to one (input layer)
        self.num_layers = 1
        # create dict to hold adjustements
        self.adjustments = {}

    def add_layer(self, shape):
        # create weights with shape specified + biases
        self.weights[self.num_layers] = np.vstack((2 * np.random.random(shape) - 1, 2 * np.random.random((1, shape[1])) - 1))
        # initialize the adjustements for these weights to zero
        self.adjustments[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, data):
        # pass data through pretrained network
        for layer in range(1, self.num_layers+1):
            data = np.dot(data, self.weights[layer-1][:, :-1]) + self.weights[layer-1][:, -1] # + self.biases[layer]
            data = self.__sigmoid(data)
        return data

    def __forward_propagate(self, data):
        # progapagate through network and hold values for use in back-propagation
        activation_values = {}
        activation_values[1] = data
        for layer in range(2, self.num_layers+1):
            data = np.dot(data.T, self.weights[layer-1][:-1, :]) + self.weights[layer-1][-1, :].T # + self.biases[layer]
            data = self.__sigmoid(data).T
            activation_values[layer] = data
        return activation_values

    def simple_error(self, outputs, targets):
        return targets - outputs

    def sum_squared_error(self, outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def __back_propagate(self, output, target):
        deltas = {}
        # delta of output Layer
        deltas[self.num_layers] = output[self.num_layers] - target

        # delta of hidden Layers
        for layer in reversed(range(2, self.num_layers)):  # All layers except input/output
            a_val = output[layer]
            weights = self.weights[layer][:-1, :]
            prev_deltas = deltas[layer+1]
            deltas[layer] = np.multiply(np.dot(weights, prev_deltas), self.__sigmoid_derivative(a_val))

        # caclculate total adjustements based on deltas
        for layer in range(1, self.num_layers):
            self.adjustments[layer] += np.dot(deltas[layer+1], output[layer].T).T

    def __gradient_descente(self, batch_size, learning_rate):
        # calculate partial derivative and take a step in that direction
        for layer in range(1, self.num_layers):
            partial_d = (1/batch_size) * self.adjustments[layer]
            self.weights[layer][:-1, :] += learning_rate * -partial_d
            self.weights[layer][-1, :] += learning_rate*1e-3 * -partial_d[-1, :]


    def train(self, inputs, targets, num_epochs, learning_rate=1, stop_accuracy=1e-5):
        error = []
        for iteration in range(num_epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]
                # pass the training set through our neural network
                output = self.__forward_propagate(x)

                # calculate the error
                loss = self.sum_squared_error(output[self.num_layers], y)
                error.append(loss)

                # calculate adjustements
                self.__back_propagate(output, y)

            self.__gradient_descente(i, learning_rate)

            # check if accuarcy criterion is satisfied
            if np.mean(error[-(i+1):]) < stop_accuracy and iteration > 0:
                break

        return(np.asarray(error), iteration+1)


def run_single_hyperparameters():

    nodes = 9  # Number of nodes in our hidden layer
    alpha = 5  # Learning Rate
    num_epochs = 1000  # Maximum number of epochs

    # create instance of a neural network
    nn = NeuralNetwork()

    # add layers (Input layer is created by default)
    nn.add_layer((2, nodes))
    nn.add_layer((nodes, 1))

    # XOR function
    training_data = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2, 1)
    training_labels = np.asarray([[0], [1], [1], [0]])

    error_rate, iteration = nn.train(training_data, training_labels, num_epochs, alpha)
    print('Error = ', np.mean(error_rate[-4:]))
    print('Epoches needed to train = ', iteration)
    error_rate = error_rate.reshape((iteration,4))

    sns.set_style("darkgrid")
    plt.plot(np.arange(iteration), error_rate[:, 0], label='[0,0]')
    plt.plot(np.arange(iteration), error_rate[:, 1], label='[0,1]')
    plt.plot(np.arange(iteration), error_rate[:, 2], label='[1,0]')
    plt.plot(np.arange(iteration), error_rate[:, 3], label='[1,1]')
    plt.plot(np.arange(iteration), np.mean(error_rate, axis=1), label='mean', color='black')
    plt.title('Error')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()

    # nn.predict(testing_data)


    return 0


def run_multiple_hyperparameters():

    # List of hyperparameters
    nodes_list = np.arange(4, 10, 1)
    alpha_list = np.arange(0.1, 15, 0.1)
    num_epochs = 100

    # Train for all hyperparameter combinations
    num_epoch_to_train = []
    for nodes in nodes_list:
        for alpha in alpha_list:
            nn = NeuralNetwork()
            nn.add_layer((2, nodes)) # Layer 2
            nn.add_layer((nodes, 1)) # Layer 3
            error_rate, iteration = nn.train(training_data, training_labels, num_epochs=num_epochs, learning_rate=alpha, stop_accuracy=1e-6)
            num_epoch_to_train.append(iteration)

    # Reshape for plotting
    z = np.asarray(num_epoch_to_train).reshape(len(nodes_list), -1)
    # np.savez('mesh.npz', alpha_list, nodes_list, z)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.set_style("darkgrid")

    for n in range(len(z)):
        plt.plot(alpha_list, z[n], label='{n} nodes'.format(n=n+4))

    ax.set_xlabel('alpha')
    ax.set_ylabel('epochs')
    ax.set_ylim([0,100])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Epochs needed to learn')
    plt.show()

    return 0


def main():

    run_multiple_hyperparameters()

    return 0




if __name__ == "__main__":
    main()
