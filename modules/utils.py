# -*- coding: utf-8 -*-

import numpy as np
from math import ceil
import matplotlib.pyplot as plt


# generator to string formatting
def gen2string(g):
  s = "[{}]" if isinstance(g, list) else "({})"
  return s.format(", ".join(map(str, g)))


# blockdiagonal matrix
def blockdiag(matrix_list):
  if len(matrix_list) == 2:
    m1 = matrix_list[0]
    m2 = matrix_list[1]
    return np.block(
      [
        [m1, np.zeros((m1.shape[0], m2.shape[1]))],
        [np.zeros((m2.shape[0], m1.shape[1])), m2],
      ]
    )
  else:
    return blockdiag([blockdiag(matrix_list[:-1]), matrix_list[-1]])


# creating 1-dimensional plots
def plot_1d(
        *functions,
        interval=[-1, 1],
        titles=None,
        xlabels=None,
        ylabels=None,
        sharey=True,
        figsize=[10, 5],
        num=500,
        samples=None
):
  num_plots = len(functions)
  fig, axs = plt.subplots(1, num_plots, sharey=sharey, figsize=figsize)
  x_ev = np.linspace(interval[0], interval[1], num=num, dtype=np.float32)[
         :, np.newaxis
         ]
  text = [[""] * num_plots if t is None else t for t in [titles, xlabels, ylabels]]
  for ax, function_list, title, xlabel, ylabel in zip(
          np.atleast_1d(axs), functions, *text
  ):
    if samples:
      for sample in samples:
        ax.plot(sample[0], sample[1], "*")

    function_list = (
      function_list if isinstance(function_list, list) else [function_list]
    )
    for f in function_list:
      ax.plot(x_ev, f(x_ev), linewidth=2)
      ax.set_title(title)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
  plt.show()


# showing images with predictions
def plot_image(idxs, images, labels, names, predictions=None, num_cols=4):
    num_rows = ceil(len(idxs) / num_cols)
    if predictions is not None:
        num_cols *= 2

    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    for i, idx in enumerate(idxs):

        plt.subplot(num_rows, num_cols, i + 1 if predictions is None else 2 * i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[idx, :, :, 0], cmap=plt.cm.binary)
        xlabel = names[labels[idx]]
        color = "blue"
        if predictions is not None:
            predicted_label = np.argmax(predictions[idx])
            if not predicted_label == labels[idx]:
                color = "red"
            xlabel += (
                f" | {names[predicted_label]} ({100 * np.max(predictions[idx]):2.0f})"
            )
        plt.xlabel(xlabel, color=color)

        if predictions is not None:
            plt.subplot(num_rows, num_cols, 2 * i + 2)
            plot_value_array(predictions[idx], labels[idx])
    plt.tight_layout()
    plt.show()


# visualizing the predictions
def plot_value_array(prediction, label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plot = plt.bar(range(10), prediction, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)

    plot[predicted_label].set_color("red")
    plot[label].set_color("blue")