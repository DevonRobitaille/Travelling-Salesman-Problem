from brute_force_algorithm import BruteForceAlgorithm
from genetic_algorithm import GeneticAlgorithm
from dynamic_plot import DynamicUpdate

import numpy as np
from random import randint
import matplotlib.pyplot as plt
import csv
import time
import sys

epsilon = 1e-6

# data: [
#   num_cities,
#   time_to_compute,
#   best_route,
#   best_dist
# ]
def writeToFileBrute(data):
    with open("../data/brute_data/brute_data_"+str(data[0])+"_V2", 'a') as file:
        outputStr = str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "\n"
        file.write(outputStr)

# data: [
#   num_cities,
#   time_to_compute,
#   populations,
#   iterations,
#   best_route,
#   best_dist
# ]
def writeToFileGenetic(data):
    with open("../data/genetic_data/genetic_data_"+str(data[0])+"_V2", 'a') as file:
        outputStr = str(data[0]) + "," + str(data[1]) + "," + str(data[2]) + "," + str(data[3]) + "," + str(data[4]) + "," + str(data[5]) + "\n"
        file.write(outputStr)

# cities: [
#   [x, y],
#   ...
# ]
def writeToFileCities(cities):
    with open("../data/cities/cities_data_"+str(len(cities))+"_V2", 'a') as file:
        outputStr = ""
        for i in range (len(cities)):            
            outputStr += str(cities[i]) + ","
        outputStr += "\n"
        file.write(outputStr)

def thread_brute(cities, show_graphs: bool, brute_force_heap: bool):
    print("Brute Force Algorithm")
    # Step 2 - Calculate Performance of the Greedy Algorithm
    start = time.time()

    brute_force_algorithm = BruteForceAlgorithm(cities, show_graphs, brute_force_heap)
    best_dist, best_order = brute_force_algorithm.findOptimalRoute()

    end = time.time()

    # Save data to file
    data = [
        len(cities),
        (end - start),
        best_order,
        best_dist
    ]
    writeToFileBrute(data)
    print("Finished Brute Force Algorithm - Time: %d, Dist: %f" % ((end-start), best_dist))
    return best_dist    

def thread_genetic(cities, brute_best_dist=0, show_graphs:bool=False):
    print("Genetic Algorithm")

    if show_graphs:
        dynamic_plot = DynamicUpdate("Genetic Algorithm", cities)

    # Step 3 - Calculate Performance of the Genetic Algorithm
    for population in range(100, 1050, 100): # for population in range(100, 1050, 100):

        iteration = 0
        genetic_algorithm = GeneticAlgorithm(cities, population)
        start = time.time()

        while genetic_algorithm.record_distance - brute_best_dist > epsilon:
            genetic_algorithm.update()
            iteration += 1

            if show_graphs:
                cities_connected = genetic_algorithm.calculateCityOrder()
                dynamic_plot.on_update(cities_connected[: , 0], cities_connected[: , 1])

        end = time.time()

        # Save data to file
        data =[
            len(cities),
            (end - start),
            population,
            iteration,
            genetic_algorithm.best_ever.order,
            genetic_algorithm.record_distance
        ]

        writeToFileGenetic(data)
        print("Finished Genetic Algorithm - Time: %d, Dist: %f, pop: %d, iter: %d" % ((end-start), genetic_algorithm.record_distance, population, iteration))
    
if __name__ == "__main__":
    num_cities = int(sys.argv[1])
    show_graphs = sys.argv[2].upper()
    brute_force_heap = sys.argv[3].upper()

    # Step 1 - Build the city network
    cities: np.array = np.zeros((num_cities, 2), dtype=np.int)
    for i in range(num_cities-1):
        cities[i] = [randint(0, 10), randint(0, 10)]
    cities[num_cities-1] = cities[0]

    writeToFileCities(cities)

    # Thread the brute force algorithm
    best_dist = thread_brute(cities, (True if show_graphs == "Y" else False), (True if brute_force_heap == "Y" else False))

    # Thread the genetic algorithm
    thread_genetic(cities, best_dist, (True if show_graphs == "Y" else False))
