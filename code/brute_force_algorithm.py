import numpy as np
import itertools
from math import dist
import matplotlib.pyplot as plt
import time

from dynamic_plot import DynamicUpdate

class BruteForceAlgorithm:
    def __init__(self, cities: np.array, show_graphs: bool, brute_force_heap: bool):
        # Cities
        self.cities: np.array = cities
        self.total_cities: int = cities.shape[0]
        self.best_dist: float = np.Infinity
        self.best_order: np.array = np.empty((self.total_cities), dtype=int)

        self.brute_force_heap = brute_force_heap

        if show_graphs:
            self.dynamic_plot = DynamicUpdate("Brute Force Algorithm", self.cities)
            self.show_graphs = show_graphs

    # Calculate the optimal route for visiting all of the cities
    # This is the function you would call to find the optimal route
    # Every other function is called from this one
    def findOptimalRoute(self):

        if self.brute_force_heap:
            # Use Heap's Algorithm
            order = np.arange(self.total_cities)
            self.heapPermutation(order, len(order))

        else:
            # Use itertools library to create all permutations (memory heavy)
            self.pythonNativePermutation()

        return self.best_dist, self.best_order
    
    # Calculate the dist to visit all of the cities
    def calcDistance(self, order: np.array):        
        sum: float = 0.0

        for i in range (order.shape[0] - 1):
            city_a_index = order[i]
            city_a = self.cities[city_a_index]

            city_b_index = order[i+1]
            city_b = self.cities[city_b_index]

            d = dist(city_a, city_b)
            sum += d

        self.dist = sum
        return self.dist

    # Return an array for the order of cities based on the best order
    def calculateCityOrder(self) -> np.array:
        # convert ga.order to cities in order
        cities_connected = np.zeros((self.total_cities, 2), dtype=np.int)
        for i in range(self.total_cities):
            cities_connected[i] = self.cities[self.best_order[i]]

        return cities_connected
    
    # Show the connected cities through a matplotlib plot
    def show(self):
        # convert ga.order to cities in order
        cities_connected = self.calculateCityOrder()

        # Plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(cities_connected[: , 0], cities_connected[: , 1], color='green')
        plt.plot(cities_connected[: , 0], cities_connected[: , 1], color='black')
        circ = plt.Circle((cities_connected[0, 0], cities_connected[0, 1]), radius=0.2, color='r')
        ax.add_patch(circ)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Brute Force Algorithm')
        plt.text(4.5, -0.5, "Dist: " + str(round(self.best_dist, 2)))
        plt.show()

    # This function calculates all of the possible permutations for orders to visit cities and sets the best order and best dist found
    def pythonNativePermutation(self):
        # Store in memory all of the possible permutations
        orders = list(itertools.permutations(np.arange(self.total_cities)))

        # loop over all permutations
        for i in range (len(self.orders)):
            order = np.asarray(self.orders[i])

            # first and last element of the order need to be the same (cyclic graph)
            if order[0] != 0 or order[len(order)-1] != len(order)-1:
                continue

            # calculate distance to all cities
            dist = self.calcDistance(order)

            # update best dist and best order if the dist is less than the previous best dist
            if (dist < self.best_dist):
                self.best_dist = dist
                self.best_order = order

                if self.show_graphs:
                    cities_connected = self.calculateCityOrder()
                    self.dynamic_plot.on_update(cities_connected[: , 0], cities_connected[: , 1])
                    time.sleep(0.2)

    # reference: https://www.geeksforgeeks.org/heaps-algorithm-for-generating-permutations/
    # My original intent was to use itertools to create all the permutation, however after 12 cities I faced program crashing memory issues
    # So I had to resort to this algorithm instead to calculate all of the possible permutations for the orders to visit cities
    def heapPermutation(self, order, size):
 
        # if size becomes 1 then prints the obtained
        # permutation
        if size == 1:
            
            # Because the trip should be cyclic (same first and last destination) the first and last element of the array should be the same
            if order[0] != 0 or order[len(order)-1] != len(order)-1:
                return

            # calculate the distance based on the order for visiting all of the cities
            dist = self.calcDistance(order)

            # is this the new best order?
            if (dist < self.best_dist):
                self.best_dist = dist
                self.best_order = order

                # Show the plot of all the connected cities
                if self.show_graphs:
                    cities_connected = self.calculateCityOrder()
                    self.dynamic_plot.on_update(cities_connected[: , 0], cities_connected[: , 1])
                    time.sleep(0.2)

            return
    
        for i in range(size):
            self.heapPermutation(order, size-1)
    
            # if size is odd, swap 0th i.e (first)
            # and (size-1)th i.e (last) element
            # else If size is even, swap ith
            # and (size-1)th i.e (last) element
            if size & 1:
                order[0], order[size-1] = order[size-1], order[0]
            else:
                order[i], order[size-1] = order[size-1], order[i]
