from setuptools import find_packages, setup

__version__ = ""

with open("coppafisher/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
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
        "dash-html-components",
        "dash_daq",
        "dask",
        "distinctipy",
        "h5py",
        "joblib",
        "magicgui",
        "matplotlib",
        "mplcursors",
        "napari[pyqt5]",
        "nd2",
        "numpy<2.1",
        "opencv-python-headless",
        "pandas",
        "psutil",
        "plotly",
        "PyQt5",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "torch>=2",
        "tqdm",
        "zarr",
    ],
    package_data={
        "coppafisher.setup": [
            "default.ini",
            "default_psf.npz",
            "dye_info_raw.npy",
        ],
        "coppafisher.omp": ["mean_spot.npy"],
        "coppafisher.plot.results_viewer": ["cell_colour.csv", "cellClassColours.json", "gene_colour.csv"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: Unix",
        "Operating System :: Windows",
    ],
)
