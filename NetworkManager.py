import copy
import pickle
import random
from threading import Thread

from PIL import Image


class DataSet:    # both data and tags are tuples
    def __init__(self, raw_data, tags):
        self.raw_data = raw_data
        self.decompiled_data = list()
        self.tags = tags

    def decompile_data(self):
        self.decompiled_data = list()
        for clust in self.raw_data:
            cluster = list()
            for raw in clust:
                cluster.append(decompile_rgb_as_01(raw))
            print(cluster)
            self.decompiled_data.append(cluster)


def decompile_rgb_with_size(raw):
    img = Image.open(raw)
    pixels = img.load()
    width = img.size[0]
    height = img.size[1]
    r = 0
    g = 0
    b = 0
    for x in range(width):
        for y in range(height):
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]

    return [r, g, b, width, height]


def decompile_rgb_as_256(raw):
    img = Image.open(raw)
    pixels = img.load()
    width = img.size[0]
    height = img.size[1]
    r = 0
    g = 0
    b = 0
    for x in range(width):
        for y in range(height):
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]

    return [(r/width)/height, (g/width)/height, (b/width)/height]


def decompile_rgb_as_01(raw):
    img = Image.open(raw)
    pixels = img.load()
    width = img.size[0]
    height = img.size[1]
    r = 0
    g = 0
    b = 0
    for x in range(width):
        for y in range(height):
            r += pixels[x, y][0]
            g += pixels[x, y][1]
            b += pixels[x, y][2]

    return [(r/width)/height/256, (g/width)/height/256, (b/width)/height/256]


class Network:
    def __init__(self, name,  depth, width, input_width, output_width, min_weight, max_weight, min_bias, max_bias):

        self.name = name
        self.depth = depth
        self.width = width
        self.nodeLayer = list(list())

        for d in range(self.depth):
            newNodes = list()
            for w in range(self.width):

                # initializing node input Weights
                if d == 0:
                    weight_amount = input_width
                else:
                    weight_amount = self.width

                weights = list()
                for weight in range(weight_amount):
                    weights.append(random.random()*(max_weight-min_weight)+min_weight)
                newNode = Node(weights, random.random()*(max_bias-min_bias)+min_bias)

                newNodes.append(newNode)

            self.nodeLayer.append(newNodes)

        #            Initialising weights once for final nodes
        self.output_layer = list()
        for i in range(output_width):
            weights = list()
            for weight in range(self.width):
                weights.append(random.random()*(max_weight-min_weight)+min_weight)

            self.output_layer.append(Node(weights, random.random()*(max_bias-min_bias)+min_bias))

    def save(self):
        with open(f'{self.name}.nw', 'wb') as file:
            pickle.dump(self, file)
            print(f'Network |{self.name}| saved')

    def parse(self, inputs):
        for node in self.nodeLayer[0]:
            node.find_value(inputs)

        for d in range(1, self.depth):
            for w in range(self.width):
                new_inputs = list()
                for n in range(len(self.nodeLayer[d-1])):
                    new_inputs.append(self.nodeLayer[d-1][n].get_value())
                self.nodeLayer[d][w].find_value(new_inputs)

        last_inputs = list()
        for n in range(len(self.nodeLayer[-1])):
            last_inputs.append(self.nodeLayer[-1][n].get_value())

        results = list()
        for i in range(len(self.output_layer)):
            self.output_layer[i].find_value(last_inputs)
            results.append(self.output_layer[i].get_value())

        return results

    def predict(self, image):

        data = decompile_rgb_as_256(image)
        results = self.parse(data)
        print(results)

        max_val = 0
        total = 0
        result = None
        for i in range(len(results)):
            if results[i] > max_val:
                result = i
                max_val = results[i]
            total += results[i]

        certainty = max_val / total

        return[result, certainty]

    def mutate(self, weight_deviance, bias_deviance, mutation_likelihood):

        for d in range(self.depth):
            for w in range(self.width):
                # bias
                if random.random() <= mutation_likelihood:
                    self.nodeLayer[d][w].bias += (random.random()*bias_deviance*2)-bias_deviance
                # weight
                for weight in self.nodeLayer[d][w].weightsIN:
                    if random.random() <= mutation_likelihood:
                        if random.randint(0, 1) == 0:
                            weight *= (random.random()+1)*weight_deviance
                        else:
                            weight /= (random.random() + 1) * weight_deviance

                        # repeat for output nodes
        for output in self.output_layer:

            if random.random() <= mutation_likelihood:
                output.bias += (random.random() * bias_deviance * 2) - bias_deviance

            for weight in output.weightsIN:
                if random.randint(0, 1) == 0:
                    weight *= (random.random() + 1) * weight_deviance
                else:
                    weight /= (random.random() + 1) * weight_deviance


def load(network_name):
    file = open(network_name, 'rb')
    network = pickle.load(file)
    print('network loaded')
    return network


def double_unit(num_in):
    num = num_in
    while num >= 100:
        num /= 10
    return num


class Node:
    def __init__(self, weights_in, bias):
        self.weightsIN = weights_in
        self.bias = bias
        self.value = None

    def find_value(self, input_vals):
        self.value = 0
        for i in range(len(input_vals)):
            self.value += input_vals[i]*self.weightsIN[i]
        self.value += self.bias
        # optionally use sigmoid function

    def get_value(self):
        return self.value


def train(start_network, mutant_amount, mutant_life, generations, data_set, weight_deviance, bias_deviance, mutation_likelihood):

    for g in range(generations): # amount of generations

        print(f"Running generation {g+1}.", end='')

        mutants = [start_network]
        for i in range(mutant_amount): # creating mutants
            mutant = copy.deepcopy(start_network)
            mutant.mutate(weight_deviance, bias_deviance, mutation_likelihood)

            mutants.append(mutant)

        accuracies_over_life = list()
        for life in range(mutant_life): # how many samples each mutant gets
            print('.', end='')
            # selecting a random image data from the dataset
            expected_result = random.randint(0, len(data_set.decompiled_data) - 1)
            inputs = data_set.decompiled_data[expected_result][random.randint(0, len(data_set.decompiled_data[expected_result])-1)]

            # threading time
            Threads = list()

            for mut in mutants: # creating a thread for each mutant with the target of parsing the data
                thread = ThreadedParse(mut, inputs)
                Threads.append(thread)
                thread.start()

            results = list()
            for thread in Threads:  # wait for all threads to finish
                thread.join()
                results.append(thread.result)

            accuracies = list()
            for result in results:
                total = 0
                for value in range(len(result)):
                    total += result[value]
                accuracies.append(result[expected_result]/total)
            accuracies_over_life.append(accuracies)
        avg_accuracies = list()
        for x in range(len(accuracies_over_life[0])):
            total = 0
            for y in range(len(accuracies_over_life)):
                total += accuracies_over_life[y][x]
            avg_accuracies.append(total/len(accuracies_over_life))

        total = 0
        closest_net = [0, 0]  # [index, accuracy]
        for i in range(len(avg_accuracies)):
            total += avg_accuracies[i]
            if avg_accuracies[i] > closest_net[1]:
                closest_net[0] = i
                closest_net[1] = avg_accuracies[i]

        avg_accuracy = total/len(avg_accuracies)

        print(f"\nCompleted\n Average accuracy: {avg_accuracy}\n highest accuracy: {closest_net[1]}({closest_net[0]})")

        start_network = copy.deepcopy(mutants[closest_net[0]])


class ThreadedParse(Thread):
    def __init__(self, network, inputs):

        Thread.__init__(self)

        self.result = None
        self.network = network
        self.inputs = inputs

    def run(self):
        self.result = self.network.parse(self.inputs)