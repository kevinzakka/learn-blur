"""Approximate a Gaussian filter with multiple box filter passes.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from blur import BoxFilter, GaussianFilter


def main(args):
  # instantiate filters
  box_filter = BoxFilter(9)
  gauss_filter = GaussianFilter(5)

  # load image
  img_path = os.path.join("./imgs/", args.name)
  img = np.asarray(Image.open(img_path)).astype("uint8")
  original_img = img.copy()

  print("Computing gaussian blur...")
  tic = time.time()
  blurred_gauss = gauss_filter.filter(img)
  toc = time.time()
  time_gauss = toc - tic

  print("Computing box blur...")
  tic = time.time()
  for _ in range(3):
    blurred_box = box_filter.filter(img)
    img = blurred_box
  toc = time.time()
  time_box = toc - tic

  fig, axes = plt.subplots(1, 3)
  labels = ["original", "box filter approx", "gaussian filter"]
  for ax, im, lab in zip(axes, [original_img, blurred_box, blurred_gauss], labels):
    ax.imshow(im)
    ax.set_xlabel(lab)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
  plt.savefig("../assets/box_blur_approx.png", format="png", dpi=300, bbox_inches='tight')
  plt.show()

  print("Box blur execution time: {}s".format(time_box))
  print("Gaussian blur execution time: {}s".format(time_gauss))
  print("Speedup: {}x".format(time_gauss/time_box))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("name", type=str,
                      help="Name of the image in assets folder.")
  args, unparsed = parser.parse_known_args()
  main(args)
