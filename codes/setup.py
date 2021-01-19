import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name = "py_countreg",
  version = "0.0.1",
  author = "WenSui Liu",
  author_email = "liuwensui@gmail.com",
  description = "Regression Models for Count Outcomes",
  long_description = long_description,
  long_description_content_type = "text/markdown",
  url = "https://github.com/statcompute/py_countreg",
  packages = setuptools.find_packages(),
  install_requires = ['numpy', 'statsmodels', 'scipy', 'pandas'],
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)

