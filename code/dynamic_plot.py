import matplotlib.pyplot as plt
import numpy as np
plt.ion()
class DynamicUpdate():
    def __init__(self, title: str, destinations):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.cities, = self.ax.plot([], [], color='black')

        plt.scatter(destinations[: , 0], destinations[: , 1], color='green')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)


    def on_update(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.cities.set_xdata(xdata)
        self.cities.set_ydata(ydata)

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
