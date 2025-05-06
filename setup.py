import os

from setuptools import find_packages, setup

__version__ = ""

with open(os.path.join(os.path.dirname(__file__), "coppafisher", "_version.py"), "r") as f:
    exec(f.read())

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    long_desc = f.read()

packages = [folder for folder in find_packages() if folder[-5:] != ".test"]  # Get rid of test packages

setup(
    name="coppafisher",
    version=__version__,
    description="coppaFISH software for Python",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Josh Duffield",
    author_email="jduffield65@gmail.com",
    maintainer="Paul Shuker",
    maintainer_email="paul.shuker@outlook.com",
    license="MIT",
    python_requires=">=3.11, <3.13",
    url="https://paulshuker.github.io/coppafisher/",
    packages=packages,
    install_requires=[
        "dash",
        "dask",
        "joblib",
        "matplotlib",
        "mplcursors",
        "napari[pyqt5]",
        "nd2",
        "numpy<2.1",
        "pandas",
        "psutil",
        "plotly",
        "PyQt5",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "torch>=2",
        "tqdm",
        "zarr<3",
    ],
    package_data={
        "coppafisher.setup": ["default.ini", "default_psf.npz", "dye_info_raw.npy"],
        "coppafisher.omp": ["mean_spot.npy"],
        "coppafisher.plot.results_viewer": ["cell_colour.csv", "cellClassColours.json", "gene_colour.csv"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: Unix",
        "Operating System :: Windows",
    ],
)
