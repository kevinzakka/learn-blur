"""Comparing box filter optimizations.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from blur import *


def main(args):
  img_path = os.path.join("./imgs/", args.name)
  img = np.asarray(Image.open(img_path)).astype("uint8")

  # generate a few odd filter sizes
  kernel_sizes = [x for x in range(3, 17, 2)]

  times = []
  for kernel_size in kernel_sizes:
    print(kernel_size)

    tic = time.time()
    _ = NaiveBoxFilter(kernel_size).filter(img)
    toc = time.time()
    time_naive = toc - tic

    tic = time.time()
    _ = BoxFilter(kernel_size).filter(img)
    toc = time.time()
    time_fast = toc - tic

    times.append([time_naive, time_fast])

  # plot
  plt.plot(kernel_sizes, [x[0] for x in times], '-o', c='C0', label='naive')
  plt.plot(kernel_sizes, [x[1] for x in times], '-o', c='C1', label='optim')
  plt.xlabel("Kernel sizes")
  plt.ylabel("Execution time (s)")
  plt.legend()
  plt.savefig("../assets/box_comparison.png", format="png", dpi=300)
  plt.show()

  speedups = [x[0]/x[1] for x in times]
  print("max speedup: {}x".format(max(speedups)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("name", type=str, help="Name of the image in assets folder.")
  args, unparsed = parser.parse_known_args()
  main(args)
