from random import uniform, randint
import numpy as np
from scipy.interpolate import interp1d
from math import dist, floor
import matplotlib.pyplot as plt

# Class that controls the functions of the genetic algorithm
# Inputs:
#   order:              np.array of shape (_) - i.e. a list of indexed points for all cities to create an order of said cities
#   population_size:    int - i.e. total number of unique DNAs for the genetic algorithm
# Outputs:
#   None
class DNA:
    def __init__(self, population_size: int = None, order: np.array = None):

        # Assume that the distance is infinity between cities and fitness is zero
        self.dist: float = np.Infinity
        self.fitness: float = 0.0

        if (isinstance(order, type(np.array))):
            self.order = order

            # Perform a quick shuffle of the order of cities
            if (uniform(0, 1) < 0.05):
                self.shuffle()

        else:
            self.order = np.arange(population_size)

            # Perform a quick shuffle
            for i in range(100):
                self.shuffle()

    # Choose two indexes and order and swap them
    # I start the random at 1 to preserve that the start city stays at the front of the order
    # I end the random at order.length-2 to keep the target city end the end of the order
    def shuffle(self):
        i = randint(1, self.order.shape[0]-2)
        j = randint(1, self.order.shape[0]-2)
        self.swap(i, j)

    # Swap two elements from order
    def swap(self, i: int, j: int):
        temp = self.order[i]
        self.order[i] = self.order[j]
        self.order[j] = temp

    # Calculate the distance between two cities
    def calcDistance(self, cities: np.array):

        city_a = cities[self.order[:-1]] #Gives cities[self.order[i]]
        city_b = cities[self.order[1:]] #Gives cities[self.order[i + 1]]

        dist = np.sqrt(np.power(city_a[:, 0] - city_b[:, 0], 2) + np.power(city_a[:, 1] - city_b[:, 1], 2))
        self.dist = np.sum(dist)
        return self.dist 

    # Map the fitness between a min and a max distance
    def mapFitness(self, min_dist: float, max_dist: float):
        map = interp1d([min_dist, max_dist], [1, 0])
        self.fitness = map(self.dist)
        return self.fitness

    # Normal the fitness to be between 0 and 1
    def normalizeFitness(self, total: float):
        self.fitness /= total

    # Crossover (mutate) the DNA to include a splice of two different orders
    def crossover(self, other_DNA):
        order1: np.array = self.order
        order2: np.array = other_DNA.order
        

        # Pick a random start and end point
        crossover_point_start = floor(randint(0, order1.shape[0] - 3))
        crossover_point_end = floor(randint(crossover_point_start+1, order1.shape[0] - 1))

        # fill the new_order with elements from order1, in the same location they were chosen from
        new_order = np.zeros((order1.shape[0]), dtype=np.int32)
        for i in range (crossover_point_start, crossover_point_end):
            new_order[i] = order1[i]

        # order 2 now only contains elements that weren't chosen by order1
        order2_mask = ~np.in1d(order2, order1[crossover_point_start: crossover_point_end])
        order2 = order2[~np.in1d(order2, order1[crossover_point_start: crossover_point_end])]

        # Add all elements in order2 that do not exist in new_order in the order they are found
        for i in range (new_order.shape[0]):
            if i >= crossover_point_start and i < crossover_point_end:
                continue

            new_order[i] = order2[0]
            order2 = order2[1:]

        # return new order
        return new_order

# Class that controls the functions of the genetic algorithm
# Inputs:
#   cites:              np.array of shape (_, 2) - i.e. a list of (x, y) coordinates of a city
#   population_total:   int - i.e. total number of unique DNAs for the genetic algorithm
# Outputs:
#   None
class GeneticAlgorithm:
    def __init__(self, cities: np.array, population_total: int):
        # Cities
        self.cities: np.array = cities
        self.total_cities: int = cities.shape[0]

        # Best path
        self.record_distance: float = np.Infinity
        self.best_ever: DNA = object

        # Population of all possible orders (order for visiting cities)
        self.population_total: int = population_total
        self.population: np.array = np.empty((self.population_total), dtype=object)
        for i in range (self.population_total):
            self.population[i] = DNA(self.total_cities)

    # The core function of the genetic algorithm
    # This will update the populations and improve the fitness of all DNA elements
    def update(self):
        min_dist: float = np.Infinity
        max_dist: float = 0

        curr_best: DNA = object

        # -- Find the best order from the population --

        for i in range (self.population.shape[0]):
            dist: float = self.population[i].calcDistance(self.cities)

            # Is this the best dist ever found?
            if dist < self.record_distance:
                self.record_distance = dist
                self.best_ever = self.population[i]

            # Is this the best this update?
            if dist < min_dist:
                min_dist = dist
                curr_best = self.population[i]

            # Is this the worst?
            if dist > max_dist:
                max_dist = dist

        # -- Update the fitness --

        # Map all the fitness values between 0 and 1
        sum: float = 0.0
        for i in range (self.population_total):
            sum += self.population[i].mapFitness(min_dist, max_dist)

        # Normalize the population to a range between 0 and 1
        for i in range (self.population_total):
            self.population[i].normalizeFitness(sum)

        # -- Update the population --

        # Create a new population
        new_population = np.empty((self.population_total), dtype=object)

        # Mutate the population
        for i in range (self.population_total):

            # pick two orders
            order1 = self.pickOne()
            order2 = self.pickOne()

            # Crossover
            new_order = order1.crossover(order2)
            new_population[i] = DNA(self.total_cities, new_order)

        # The new population
        self.population = new_population
    
    # This function random selects an index for which DNA to use for the crossover
    # It completes this by seleting a random float between 0 and 1
    # Then we continuously subtract from r the fitness of a DNA
    # The idea is that whichever DNA trips the threshold should have had a large enough impact that their DNA is worth using for the next generation
    def pickOne(self) -> DNA:
        index: int = 0

        # Pick a random number between 0 and 1
        r: float = uniform(0, 1)

        while r > 0:
            r -= self.population[index].fitness
            # Move to the next index
            index += 1

        # Go back one (due to the while loop)
        index -= 1

        return self.population[index]

    # Return the cities tuple array re-ordered based on the best recorded order for said cities
    def calculateCityOrder(self) -> np.array:
         # convert ga.order to cities in order
        cities_connected = np.empty((self.total_cities, 2), dtype=np.int)
        for i in range(self.total_cities):
            cities_connected[i] = self.cities[self.best_ever.order[i]]

        return cities_connected

    # A function to display how all the cities are connected in a matplotlib plot
    def show(self):
        # convert ga.order to cities in order
        cities_connected = np.empty((self.total_cities, 2), dtype=np.int)
        for i in range(self.total_cities):
            cities_connected[i] = self.cities[self.best_ever.order[i]]

        # Plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(cities_connected[: , 0], cities_connected[: , 1], color='green')
        plt.plot(cities_connected[: , 0], cities_connected[: , 1], color='black')
        circ = plt.Circle((cities_connected[0, 0], cities_connected[0, 1]), radius=0.2, color='r')
        ax.add_patch(circ)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Genetic Algorithm')
        plt.text(4.5, -0.5, "Dist: " + str(round(self.record_distance, 2)))
        plt.show()

