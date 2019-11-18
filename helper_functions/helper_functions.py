# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# print 1-dimensional functions
def plot_1d(*functions, interval=[-1,1], titles=[''], xlabels=[''], 
             ylabels=[''], sharey=True, figsize=[10,5], num=500, samples=None):
  num_plots = len(functions)
  fig, axs = plt.subplots(1, num_plots, sharey=sharey, figsize=figsize)
  x_ev=np.linspace(interval[0],interval[1],num=num,
                    dtype=np.float32)[:,np.newaxis]
  for obj in [titles,xlabels,ylabels]:
    obj+=['']*(num_plots-len(obj))
  for ax, function_list, title, xlabel, ylabel in zip(np.atleast_1d(axs), 
                                                      functions, titles, 
                                                      xlabels, ylabels):
    for f in np.atleast_1d(function_list):
      ax.plot(x_ev,f(x_ev),linewidth=2)
      if samples:
        for sample in samples:
          ax.plot(sample[0],sample[1],'*')
      ax.set_title(title)   
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)   
  plt.show()

# return blockdiagonal matrix
def blockdiag(matrix_list):
  if len(matrix_list)==2:
    m1 = matrix_list[0]
    m2 = matrix_list[1]
    return np.block([[m1,np.zeros((m1.shape[0],m2.shape[1]))],
                     [np.zeros((m2.shape[0],m1.shape[1])),m2]])
  else:
    return blockdiag([blockdiag(matrix_list[:-1]),matrix_list[-1]])
