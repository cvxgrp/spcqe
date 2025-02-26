# spcqe
Smooth (multi-) periodic consistent quantile estimation. We attempt to follow the sklearn "fit/transform" API, and the main class inherets `TransformerMixin` and `BaseEstimator` from `sklearn.base`.


## Installation

The package is available on both PyPI and conda-forge.

pip installation:

```
pip install spcqe
```

conda installation:

```
conda install conda-forge::spcqe 
```

You may also clone the repository to your local machine and install with pip by navigating to the project directory and running:

```
pip install .
```

If working on the files in this package (i.e. fixing bugs or adding features), it useful to install in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```
pip install -e .
```


## Usage

```
from spcqe.quantiles import SmoothPeriodicQuantiles

y1 = ... # some hourly data with daily, weekly, and yearly periodic statistics
P1 = int(365*24)
P2 = int(7*24)
P3 = int(24)
K = 3
l = 0.1
spq = SmoothPeriodicQuantiles(K, [P1, P2, P3], weight=l)
spq.fit(y1)
```

## Examples 

Many examples Jupyter notebooks are available in the `notebooks` folder.

## Acknowledgement 

This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Solar Energy Technologies Office Award Number 38529, "PVInsight".
