"""Common blurring filters.
"""

import abc

import numpy as np

from numba import njit, prange


class Filter(abc.ABC):
  """Base filter abstraction.
  """
  def _pad(self, x, k, mode="edge"):
    """Evenly pad an image on all sides.

    Args:
      x (ndarray): The image to pad.
      k (int): The amount to pad on each side.
      mode (str): The type of padding. Can be one of `edge` or
        `constant`.
    """
    c = x.shape[-1]
    pad = [(k,), (k,)]
    per_chan = [np.pad(x[:, :, chan], pad, mode=mode) for chan in range(c)]
    return np.stack(per_chan, axis=2)

  @abc.abstractmethod
  def filter(self, x):
    """Apply the filter to an input image x.

    Args:
      x: A numpy array of shape (H, W, C) where C
        can be either 1 or 3.

    Returns:
      out: the blurred input of type `uint8`.
    """
    raise NotImplementedError


class NaiveBoxFilter(Filter):
  """A naive box blur filter.

  This runs in O(h*w*k^2), i.e. it runs quadratic
  in filter size.
  """
  def __init__(self, kernel_size):
    """Constructor.

    Args:
      kernel_size (int): The size of the box. This should be
        an odd number.
    """
    super().__init__()
    self.kernel_size = kernel_size
    self.numel = kernel_size * kernel_size

  def filter(self, x):
    if x.ndim == 2:
      x = x[:, :, np.newaxis]
    h, w, c = x.shape
    k = (self.kernel_size - 1) // 2
    x_pad = self._pad(x, k)
    out = np.empty_like(x, dtype="float64")
    for row in range(h):
      for col in range(w):
        cumsum = np.zeros(c)
        for i in range(-k, k+1):
          for j in range(-k, k+1):
            cumsum += x_pad[row+i+k, col+j+k, :]
        out[row, col, :] = cumsum
    out = (out / self.numel).astype("uint8")
    return out.squeeze()


class BoxFilter(Filter):
  """A faster box blur filter.

  This runs in O(h*w), i.e. it runs independent
  of kernel size.
  """
  def __init__(self, kernel_size):
    """Constructor.

    Args:
      kernel_size (int): The size of the box. This should be
        an odd number.
    """
    super().__init__()
    self.kernel_size = kernel_size
    self.numel = kernel_size * kernel_size

  @staticmethod
  @njit(parallel=True)
  def _loop(h, w, c, k, x_pad, out):
    for row in prange(h):
      cumsum = np.zeros(c)
      for i in range(-k, k+1):
        cumsum += x_pad[row+k, i+k, :]
      out[row, 0] = cumsum
      for col in range(1, w):
        cumsum = cumsum - x_pad[row+k, col-1-k+k, :] + x_pad[row+k, col+2*k, :]
        out[row, col, :] = cumsum
    return out

  def _filter1d(self, x, k):
    """Perform a horizontal motion blur.
    """
    h, w, c = x.shape
    x_pad = self._pad(x, k)
    out = np.empty_like(x, dtype="float64")
    out = self._loop(h, w, c, k, x_pad, out)
    return out

  def filter(self, x):
    if x.ndim == 2:
      x = x[:, :, np.newaxis]
    h, w, c = x.shape
    k = (self.kernel_size - 1) // 2
    horiz = self._filter1d(x, k)
    vert = self._filter1d(horiz.transpose((1, 0, 2)), k)
    out = (vert.transpose((1, 0, 2)) / self.numel).astype("uint8")
    return out.squeeze()


class GaussianFilter(Filter):
  """A Gaussian blur filter.
  """
  def __init__(self, sigma, truncate=3.0):
    """Constructor.

    Args:
      sigma (float or tuple): The standard deviation of the Gaussian
        kernel for each axis. If a single float is provided, the same
        standard deviation is used for both axes. The higher this number,
        the stronger the blur.
      truncate (float): After how many standard deviations to truncate
        the Gaussian kernel.

    References:
      [1]: https://blog.demofox.org/2015/08/19/gaussian-blur/
      [2]: https://github.com/scipy/scipy/blob/f2ec91c4908f9d67b5445fbfacce7f47518b35d1/scipy/ndimage/filters.py#L211
    """
    super().__init__()

    assert sigma >= 0., "[!] Standard deviation must be positive."

    self.sigma = float(sigma)
    self.var = self.sigma * self.sigma
    self.truncate = truncate

    self.k = self._compute_kernel_radius()
    self.kernel = self._sample_kernel_gaussian_function()

  def _compute_kernel_radius(self, trunc=True):
    """Compute the kernel radius given a standard deviation.
    """
    if trunc:  # from [2]
      radius = int(self.truncate * self.sigma + 0.5)
    else:  # from [1]
      thresh = 0.005
      radius = int(np.floor(1.0 + 2.0*np.sqrt(-2.0*self.var*np.log(thresh)))) + 1
    return radius

  def _sample_kernel_pascal_triangle(self):
    """Sample kernel weights using Pascal's Triangle approximation.
    """
    from scipy.linalg import pascal
    kernel = pascal(2*self.k+1, kind='lower')[-1]
    kernel = kernel / np.sum(kernel)
    return kernel

  def _sample_kernel_gaussian_function(self):
    """Sample kernel weights by evaluating the Gaussian function at evenly-spaced integers.
    """
    x = np.linspace(-self.k, self.k, 2*self.k+1)
    kernel = np.exp(-0.5 * x * x / self.var)
    kernel = kernel / np.sum(kernel)
    return kernel

  @staticmethod
  @njit(parallel=True)
  def _loop(h, w, c, k, kernel, x_pad, out):
    for row in prange(h):
      for col in range(w):
        cumsum = np.zeros(c)
        for i in range(-k, k+1):
          cumsum += kernel[i+k] * x_pad[row+k, col+i+k, :]
        out[row, col, :] = cumsum
    return out

  def _filter1d(self, x):
    """Filter the image along the horizontal axis.
    """
    h, w, c = x.shape
    x_pad = self._pad(x, self.k)
    out = np.empty_like(x, dtype="float64")
    out = self._loop(h, w, c, self.k, self.kernel, x_pad, out)
    return out

  def filter(self, x):
    if x.ndim == 2:
      x = x[:, :, np.newaxis]
    h, w, c = x.shape
    horiz = self._filter1d(x)
    vert = self._filter1d(horiz.transpose((1, 0, 2)))
    out = vert.transpose((1, 0, 2)).astype("uint8")
    return out.squeeze()


# TODO figure out why numba complains about this
class MedianFilter(Filter):
  """A median blur filter.
  """
  def __init__(self, kernel_size):
    """Constructor.

    Args:
      kernel_size (int): The size of the box. This should be
        an odd number.
    """
    super().__init__()
    self.kernel_size = kernel_size

  def filter(self, x):
    if x.ndim == 2:
      x = x[:, :, np.newaxis]
    h, w, c = x.shape
    k = (self.kernel_size - 1) // 2
    x_pad = self._pad(x, k)
    out = np.empty_like(x, dtype=np.uint8)
    for row in range(h):
      for col in range(w):
        for chan in range(c):
          cumsum = []
          for i in range(-k, k+1):
            for j in range(-k, k+1):
              cumsum.append(x_pad[row+i+k, col+j+k, chan])
          out[row, col, chan] = int(np.median(cumsum))
    return out.squeeze()


class BilateralFilter(Filter):
  """A bilateral smoothing filter.
  """
  def __init__(self, sigma_color, sigma_spatial, truncate):
    """Constructor.

    Args:
      sigma_color (float or tuple):
      sigma_spatial (float or tuple):
      truncate (float or tuple): After how many standard deviations
        to truncate the Gaussian kernel for both color and spatial
        kernels.
    """
    super().__init__()

    assert sigma_color >= 0., "[!] Standard deviations must be positive."
    assert sigma_spatial >= 0., "[!] Standard deviations must be positive."

    self.sigma_color = float(sigma_color)
    self.sigma_spatial = float(sigma_spatial)
    self.truncate = float(truncate)

    self.k_color = self._compute_kernel_radius()

  def _gaussian_function(self, x, mu, sigma):
    """
    """
    pass

  def _compute_kernel_radius(self, sigma, trunc=True):
    """Compute the kernel radius given a standard deviation.
    """
    return int(self.truncate * sigma + 0.5)

  def _sample_kernel_discrete_gaussian_function(self):
    """Sample kernel weights by evaluating the Gaussian function at evenly-spaced integers.
    """
    x = np.linspace(-self.k, self.k, 2*self.k+1)
    kernel = np.exp(-0.5 * x * x / self.var)
    kernel = kernel / np.sum(kernel)
    return kernel

  def filter(self, x):
    if x.ndim == 2:
      x = x[:, :, np.newaxis]
    h, w, c = x.shape
    return