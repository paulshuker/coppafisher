## Coppafish

<div class="grid cards" markdown>

- :material-checkbox-multiple-blank: __Zarr__ for image compression
- :material-fast-forward: __PyTorch__ for GPU/CPU acceleration
- :material-eye: __Napari__ for 3D visualisation
- :material-web: __Dash__ for web interaction

</div>

Coppafish is an open source data analysis software for COmbinatorial Padlock-Probe-Amplified Fluorescence In Situ
Hybridization (coppafish) datasets. A series of 3D microscope images are arranged into tiles, rounds and channels. For
each sequencing round, every considered gene is fluoresced by a dye. By the end of all rounds, each gene has a unique,
barcode-like sequence of dyes, called the gene code. For more details about coppafish's methodology, see the
[overview](overview.md). See [installation](#installation) on how to install our software, and [usage](basic_usage.md) to
run coppafish on your dataset. Some vocabulary might be unfamiliar, please see the [glossary](glossary.md) for
reference.

<figure markdown="span">
  ![Image title](images/coppafish_simple_schematic.png){ width="400" }
  <figcaption>Gene calling on a tile.</figcaption>
</figure>

## Installation

### Prerequisites

* Windows or Linux operating system. MacOS is not tested.
* At least 48GB of RAM for tile sizes `58x2048x2048`.
* Python version 3.9 or 3.10.
* [Git](https://git-scm.com/).

### Environment

Install coppafish software from within an environment. This can be a `venv` or `conda` (recommended) environment.

#### Conda

For `conda`, build an environment by doing:
```terminal
conda create -n coppafish python=3.10
conda activate coppafish
```

### Install

Our latest coppafish release can be cloned locally
```terminal
git clone --depth 1 https://github.com/paulshuker/coppafish
```

or get a specific version. For example, version 1.0.0 by

```terminal
git clone --depth 1 --branch 1.0.0 https://github.com/paulshuker/coppafish
```

then install package dependencies and coppafish by

```terminal
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install .
```

## Updating

Coppafish will not automatically install updates. But, you will see a warning at the start of a pipeline if a new 
online version is available.

To update version, follow all [install](#install) instructions again while inside of your coppafish conda environment.

You can verify your install by running `#!terminal pip show coppafish` in the coppafish environment to check you have the 
latest version.

## Changing software version

If you wish to know what old output files should be deleted when migrating from one software version to another, run 
in the python terminal
```python
from coppafish import CompatibilityTracker

track = CompatibilityTracker()
track.check(X, Y)
```

where X is the old coppafish version, Y is the new coppafish version. This works for coppafish versions 0.10.7 and 
above. For example, to find what files to delete when migrating from 0.10.7 to 1.0.0, run
```python
from coppafish import CompatibilityTracker

track = CompatibilityTracker()
track.check("0.10.7", "1.0.0")
```
