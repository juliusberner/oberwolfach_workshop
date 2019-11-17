# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# plot 1-dimensional functions
def plot_1d(*functions, interval=[-1,1], titles=[''], xlabels=[''], 
             ylabels=[''], sharey=True, figsize=[10,5], num=500):
  num_plots = len(functions)
  fig, axs = plt.subplots(1, num_plots, sharey=sharey, figsize=figsize)
  x_ev=np.linspace(interval[0],interval[1],num=num, dtype=np.float32)[:,np.newaxis]
  for obj in [titles,xlabels,ylabels]:
    obj+=['']*(num_plots-len(obj))
  for ax, function_list, title, xlabel, ylabel in zip(np.atleast_1d(axs), 
                                                      functions, titles, 
                                                      xlabels, ylabels):
    for f in np.atleast_1d(function_list):
      ax.plot(x_ev,f(x_ev),linewidth=2)
      ax.set_title(title)   
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)   
  plt.show()

