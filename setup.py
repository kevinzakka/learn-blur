from setuptools import setup

setup(
  name="blur",
  version="0.0.1",
  description="Learning about various image blur techniques.",
  author="Kevin",
  python_requires=">=3.6",
  install_requires=[
    "numpy",
    "numba",
    "matplotlib",
  ],
)