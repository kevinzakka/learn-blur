# Notes

**Why do filters have odd sizes?**

So we can center the kernel at the pixel.

**Why does making a filter separable make it faster?**

Let's take a simple numerical example. Suppose our image is 100x100. Suppose our kernel is 10x10. We need to do a computation for all the pixels in the image, i.e. 100x100 = 10,000 pixels. For each pixel, we need to do a multiplication for every pixel in the filter, i.e. 10x10 = 100. So we have a total of 10,000 x 100 = 1,000,000 multiplications. In contrast, if we do a vertical convolution with a 1D filter of size 10 followed by a horizontal convolution with a 1D filter of size 10, we have: first, we're going to do computations for every pixel in the image so 10,000. For each pixel, we do 10 operations. So a total of 100,000 for the vertical so a total of 2*100K = 200,000 operations for the separable case. That's 5 times less multiplications. This becomes even more drastic as the filter size increases.

**Why do multiple passes of a box blur filter approximate a Gaussian filter?**

From probability theory, we know that the probability density of a sum of independent random variables is obtained by the convolution of the probability densitites of the respective variables. And from the Central Limit Theorem, the sum of independent variables tends towards a normal distribution. So n convolutions of the box filter with itself produces a filter whose impulse response tends towards a Gaussian function as n tends to infinity.

**What kind of images would require a higher spatial sigma, and what kind a higher range sigma?**