import argparse
import csv
import numpy as np
from numpy import random


class Perceptron:
    def __init__(self):
        self.value = None
        self.weights = []
        self.bias = np.random.normal(0, 0.1)
        self.child_perceptrons = []
        self.parent_perceptrons = []

    def compute_value(self, index):
        weighted_sum = np.dot([parent.value for parent in self.parent_perceptrons], [parent.weights[index] for parent in self.parent_perceptrons]) + self.bias
        self.value = 1 / (1 + np.exp(-weighted_sum))

    def compute_endvalue(self, index):
        weighted_sum = np.dot([parent.value for parent in self.parent_perceptrons], [parent.weights[index] for parent in self.parent_perceptrons]) + self.bias
        self.value = weighted_sum


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_first_layer(self):
        return self.layers[0]

    def get_last_layer(self):
        return self.layers[-1]


def initNN(arg, y, num_of_features):
    start_perc = []
    temp_perc = []
    end_perc = []

    neural_network = NeuralNetwork()
    neural_network.add_layer(start_perc)

    layer_values = [int(layer) for layer in arg.split("s") if layer]

    for _ in range(num_of_features):
        start_perc.append(Perceptron())
    for _ in y[0]:
        end_perc.append(Perceptron())

    total_weights = 0
    previous_layer_neurons = num_of_features

    for neurons in layer_values:
        total_weights += previous_layer_neurons * neurons
        previous_layer_neurons = neurons

    total_weights += previous_layer_neurons

    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, total_weights)

    counter = 0
    current_perc = start_perc
    for neurons in layer_values:
        temp_perc = []
        for _ in range(neurons):
            temp = Perceptron()
            temp.parent_perceptrons.extend(current_perc)
            for parent in current_perc:
                parent.weights.append(s[counter])
                parent.child_perceptrons.append(temp)
                counter += 1
            temp_perc.append(temp)
        current_perc = temp_perc
        neural_network.add_layer(current_perc)

    for end in end_perc:
        end.parent_perceptrons.extend(current_perc)
        for parent in current_perc:
            parent.weights.append(s[counter])
            counter += 1
            parent.child_perceptrons.append(end)
    neural_network.add_layer(end_perc)
    return start_perc, end_perc, neural_network


def propagate(start_perc, end_perc, x):
    for i, value in enumerate(x):
        start_perc[i].value = float(value)

    current_layer = start_perc
    while True:
        next_layer = []
        for perceptron in current_layer:
            for child in perceptron.child_perceptrons:
                if child not in next_layer:
                    next_layer.append(child)
        i = 0
        if next_layer == end_perc:
            for perceptron in next_layer:
                perceptron.compute_endvalue(i)
                i += 1
            break
        for perceptron in next_layer:
            perceptron.compute_value(i)
            i += 1
        current_layer = next_layer

    return [perc.value for perc in end_perc]


def calculate_error(y_est, y):
    assert len(y_est) == len(y)
    error_sum = 0
    error_sum = np.mean((np.array(y_est) - np.array(y)) ** 2)
    return error_sum / len(y_est)


def evaluate(Neural_Network, x_train, y):
    first_layer = Neural_Network.get_first_layer()
    last_layer = Neural_Network.get_last_layer()
    NN_of_x = []
    for instance in x_train:
        NN_of_x.append(propagate(first_layer, last_layer, instance))
    return calculate_error(NN_of_x, y)


def load_csv(filename):
    x = []
    y = []
    num_features = 0

    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        header = csv_reader.fieldnames

        for row in csv_reader:
            features = [float(row[column]) for column in header[:-1]]
            target = float(row[header[-1]])
            x.append(features)
            y.append([target])

        num_features = len(header) - 1

    return x, y, num_features

def crossover(parent1, parent2):
    child = NeuralNetwork()
    for layer1, layer2 in zip(parent1.layers, parent2.layers):
        child_layer = []
        for perc1, perc2 in zip(layer1, layer2):
            child_perc = Perceptron()
            child_perc.weights = [(w1 + w2) / 2 for w1, w2 in zip(perc1.weights, perc2.weights)]
            child_perc.bias = (perc1.bias + perc2.bias) / 2
            child_layer.append(child_perc)
        child.add_layer(child_layer)
    for layer_index in range(1, len(child.layers)):
        current_layer = child.layers[layer_index]
        previous_layer = child.layers[layer_index - 1]

        for perceptron in current_layer:
            perceptron.parent_perceptrons.extend(previous_layer)
            for parent in previous_layer:
                parent.child_perceptrons.append(perceptron)

    return child

def choose(Neural_Networks, elitism_number, mutation_rate, mutation_scale, x_train, y_train):
    Neural_Networks.sort(key=lambda x: x[1])

    elite_population = Neural_Networks[:elitism_number]
    Neural_Networks = Neural_Networks[elitism_number:]
    fitness_values = np.array([instance[1] for instance in Neural_Networks[elitism_number:]])

    total_fitness = np.sum(fitness_values)

    new_population = []

    def select_parent():
        pick = random.uniform(0, total_fitness)
        current = 0
        for nn, fitness in Neural_Networks[elitism_number:]:
            current += fitness
            if current > pick:
                return nn, fitness

    parent1, fitness1 = select_parent()
    parent2, fitness2 = select_parent()
    parent3, fitness3 = select_parent()

    parents = [(parent1, fitness1), (parent2, fitness1), (parent3, fitness1)]
    parents.sort(key=lambda x: x[1])

    parent1, fitness1 = parents[-2]
    parent2, fitness2 = parents[-1]

    worst_parent = parents[0]
    Neural_Networks.remove(worst_parent)
    total_fitness -= worst_parent[1]
    child1 = crossover(parent1, parent2)

    mutate(child1, mutation_scale, mutation_rate)
    fitness_child1 = evaluate(child1, x_train, y_train)

    new_population.append((child1, fitness_child1))

    new_population.extend(elite_population)
    new_population.extend(Neural_Networks)

    return new_population

def mutate(neural_network, mutation_scale, mutation_rate):
    for layer in neural_network.layers:
        for perceptron in layer:
            if random.random() < mutation_rate:
                perceptron.weights = np.array(perceptron.weights)
                perceptron.weights += np.random.normal(0, mutation_scale, perceptron.weights.shape)
                perceptron.bias += np.random.normal(0, mutation_scale)

def genetic_alg(elitism_number, mutation_rate, mutation_scale, iteration_number, Neural_Networks, x_train,
                y_train, x_test, y_test):
    counter = 0
    while counter < iteration_number:
        new_population = choose(Neural_Networks, elitism_number, mutation_rate, mutation_scale, x_train, y_train)
        Neural_Networks = new_population

        best_network, best_mse = min(Neural_Networks, key=lambda x: x[1])

        if (counter + 1) % 2000 == 0:
            print("[Train error @{}]: {:.6f}".format(counter + 1, best_mse))

        counter += 1

    NN_of_x_test = [propagate(best_network.get_first_layer(), best_network.get_last_layer(), instance) for instance in
                    x_test]
    test_error = calculate_error(NN_of_x_test, y_test)
    print("[Test error]: {:.6f}".format(test_error))


def create_population(filename_train, filename_test, layers, popsize, elitism_number, mutation_rate, mutation_scale,
                      iteration_number):
    x_train, y_train, num_of_features = load_csv(filename_train)
    x_test, y_test, num_of_features = load_csv(filename_train)
    Neural_Networks = []

    for i in range(popsize):
        start_perc, end_perc, neural_network = initNN(layers, y_train, num_of_features)
        NN_of_x = []
        NN_of_x = [propagate(start_perc, end_perc, instance) for instance in x_train]
        error = calculate_error(NN_of_x, y_train)
        Neural_Networks.append((neural_network, 1/error))

    genetic_alg(elitism_number, mutation_rate, mutation_scale, iteration_number, Neural_Networks, x_train,
                y_train, x_test, y_test)


def main():
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    parser.add_argument('--nn', type=str, required=True)
    parser.add_argument('--popsize', type=int)
    parser.add_argument('--elitism', type=int)
    parser.add_argument('--p', type=float)
    parser.add_argument('--K', type=float)
    parser.add_argument('--iter', type=int)

    args = parser.parse_args()

    create_population(args.train, args.test, args.nn, args.popsize, args.elitism, args.p, args.K, args.iter)

if __name__ == '__main__':
    main()
