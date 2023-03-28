import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np


def plotLine2d(y: Union[List[float], np.ndarray],
               x: Union[List[float], np.ndarray] = None,
               y_name: str = "Y axis",
               x_name: str = "X axis",
               plot_name: str = "Plot"):
    if x is None:
        if isinstance(y, list):
            x = list(range(len(y)))
        elif isinstance(y, np.ndarray):
            x = list(range(y.shape[0]))
        else:
            raise ValueError("Input y array is of unsupported type")
    
    fig = plt.figure()
    ax = plt.axes()

    plt.plot(x, y)
    plt.title(plot_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.show()