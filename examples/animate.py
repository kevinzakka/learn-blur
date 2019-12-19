"""Cute blur animation.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from PIL import Image

from blur import BoxFilter


if __name__ == "__main__":
  img_path = "./imgs/cat-crop.png"
  img = np.asarray(Image.open(img_path)).astype("uint8")

  h, w, c = img.shape
  blurred = BoxFilter(11).filter(img)

  out = np.zeros_like(img, dtype="uint8")
  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(img)
  canvas = axes[1].imshow(out)
  for ax in axes:
    ax.axis('off')

  def gen_func():
    for row in range(h):
      yield row,

  def update_func(row):
    row = row[0]
    for col in range(w):
      out[row, col, :] = blurred[row, col, :]
    canvas.set_data(out)
    return canvas,

  ani = animation.FuncAnimation(fig, update_func, gen_func, interval=2, blit=True, save_count=200)
  ani.save('../assets/blur_ani.gif', writer='imagemagick')
