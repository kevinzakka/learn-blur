"""Comparing various smoothing filters.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from blur import *


def main(args):
  if args.filter == "box":
    blur_filter = BoxFilter(args.kernel_size)
  elif args.filter == "gaussian":
    blur_filter = GaussianFilter(args.std_dev, args.truncate)
  elif args.filter == "bilateral":
    pass
  elif args.filter == "median":
    blur_filter = MedianFilter(args.kernel_size)
  else:
    raise ValueError("{} filter not supported.".format(args.filter))

  print("loading image...")
  img_path = os.path.join("./imgs/", args.name)
  img = np.asarray(Image.open(img_path)).astype("uint8")

  print("applying {} blur...".format(args.filter))
  blurred = blur_filter.filter(img)

  fig, axes = plt.subplots(1, 2)
  for ax, im in zip(axes, [img, blurred]):
    ax.imshow(im)
    ax.axis('off')
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("name", type=str,
                      help="Name of the image in assets folder.")
  parser.add_argument("--filter", type=str, default="gaussian",
                      help="Which filter to use.")
  parser.add_argument("--std_dev", "--sigma", type=float, default=1.0,
                      help="The standard deviation of the Gaussian filter.")
  parser.add_argument("--kernel_size", "--filter_size", type=int, default=3,
                      help="The size of the filter.")
  parser.add_argument("--truncate", type=float, default=3.0,
                      help="The truncation for the Gaussian filter.")
  args, unparsed = parser.parse_known_args()
  main(args)
