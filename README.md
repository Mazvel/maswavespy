# MASWavesPy

MASWavesPy (`maswavespy`) is a Python package for processing and inverting MASW data, developed at the Faculty of Civil and Environmental Engineering, University of Iceland. 

### Table of contents
- [About MASWavesPy](#about-maswavespy)
  - [Referencing MASWavesPy](#referencing-maswavespy)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
- [Installation](#installation)
   - [General installation using pip](#general-installation-using-pip)
   - [Recommendations](#recommendations)
   - [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide) (for Windows)
  - [Setup and create a virtual environment (recommended)](#setup-and-create-a-virtual-environment-recommended)
  - [Install MASWavesPy](#install-maswavespy)
  - [Test MASWavesPy](#test-maswavespy)
  - [Deactivate the virtual environment](#deactivate-the-virtual-environment)
- [Known Issues](#known-issues)

## About MASWavesPy

The `maswavespy` package consists of four primary modules: `wavefield`, `dispersion`, `combination` and `inversion`, and two supplementary modules: `dataset` and `select_dc`. 

The `wavefield` module provides methods to import recorded shot gathers as `RecordMC` objects. The phase shift method (1) is used to transform each shot gather into the frequency-phase velocity domain. The `dataset` module can be used to import a set of shot gathers in the form of a `Dataset` object through a .csv file. 

The `dispersion` module, along with the supplementary `select_dc` module, provides methods for visualization of the phase velocity spectrum and dispersion curve (DC) identification using a GUI (Graphical User Interface). An `ElementDC` object stores the frequency-phase velocity domain representation of a given `RecordMC` and the corresponding DC (referred to as an elementary DC). 

The `combination` module provides methods to combine elementary DCs obtained from multiple shot gathers into a composite DC (2) (a `CombineDCs` object) and to assess and view the spread in the dispersion data, either as a function of frequency or wavelength. A `Dataset` object can contain multiple pairs of `RecordMC` and `ElementDC` objects (one pair for each shot gather) and provides routines for initializing a `CombineDCs` for the set of records or a particular subset of records. 

The `inversion` module provides methods to evaluate the shear wave velocity profile of the tested site. The inversion methods, along with routines for post-processing of the inversion results, are defined on an `InvertDC` object that is initialized using an experimental DC. The fast delta matrix algorithm (3) is used for forward computations and a Monte-Carlo global search algorithm (4) for searching the solution space for the optimal set of model parameters. 

A more comprehensive description is provided in (5). 

### Referencing MASWavesPy
Referencing the MASWavesPy package and a paper related to its development is highly appreciated. 

> Olafsdottir, E.A., Bessason, B., Erlingsson, S., Kaynia, A.M. (2024). A Tool for Processing and Inversion of MASW Data and a Study of Inter-Session Variability of MASW. Accepted for publication in _Geotechnical Testing Journal_ (in press).

### License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Acknowledgements
This work was supported by the Icelandic Research Fund [grant numbers 206793-052 and 218149-051], the University of Iceland Research Fund, the Icelandic Road and Coastal Administration and the Energy Research Fund of the National Power Company of Iceland.

> (1) Park, C.B., Miller, R.D., Xia, J. (1998). Imaging dispersion curves of surface waves on multi-channel record. In _SEG Technical Program Expanded Abstracts 1998_, New Orleans, Louisiana, pp. 1377–1380. [https://doi.org/10.1190/1.1820161](https://doi.org/10.1190/1.1820161)
> 
> (2) Olafsdottir, E.A., Bessason, B., Erlingsson, S. (2018a). Combination of dispersion curves from MASW measurements. _Soil Dynamics and Earthquake Engineering_, 113, pp. 473–487. [https://doi.org/10.1016/j.soildyn.2018.05.025](https://doi.org/10.1016/j.soildyn.2018.05.025)
> 
> (3) Buchen, P.W., Ben-Hador, R. (1996). Free-mode surface-wave computations. _Geophysical Journal International_, 124(3), pp. 869–887. [https://doi.org/10.1111/j.1365-246X.1996.tb05642.x](https://doi.org/10.1111/j.1365-246X.1996.tb05642.x)
> 
> (4) Olafsdottir, E.A., Erlingsson, S., Bessason, B. (2020). Open-Source MASW Inversion Tool Aimed at Shear Wave Velocity Profiling for Soil Site Explorations, _Geosciences_, 10(8), 322. [https://doi.org/10.3390/geosciences10080322](https://doi.org/10.3390/geosciences10080322)
> 
> (5) Olafsdottir, E.A., Bessason, B., Erlingsson, S., Kaynia, A.M. (2024). A Tool for Processing and Inversion of MASW Data and a Study of Inter-Session Variability of MASW. Accepted for publication in _Geotechnical Testing Journal_ (in press).

## Installation
A [Quick Start Guide](#quick-start-guide) describing the recommended workflow for Windows users is provided below.

### General installation using pip
The MASWavesPy package is installed using pip. 

`pip install maswavespy`

Wheels for Windows, Linux and Mac distributions can also be downloaded from [PyPI](https://pypi.org/project/maswavespy/#files).


### Recommendations
We recommend to install the MASWavesPy package into an isolated Python environment. If using Anaconda, create a virtual environment using [conda create](https://docs.conda.io/projects/conda/en/latest/commands/create.html). Alternatively, [virtualenv](https://docs.python.org/3/library/venv.html) can be used to install this package into an isolated Python environment. [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is a tool to simplify the creation and management of local virtualenvs.

The use of a Python IDE (Integrated Development Environment) is strongly recommended for using MASWavesPy (as opposed to running commands in the Windows terminal/cmd environment). 

MASWavesPy is developed using the [Anaconda distribution](https://www.anaconda.com/). Hence Anaconda and the Spyder IDE (included with Anaconda) are recommended for running the Quick Start Guide. 

### Requirements

To build the package on Windows you need Microsoft C++ Build Tools. You can download an installer from Microsoft at this [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Otherwise you will see an error:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
For more information you can view this Stackoverflow [answer](error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/)

This is required because the package uses [Cython](https://cython.org/) for some of its calculations.

## Quick Start Guide
**Applies for Windows Users.**

### Setup and create a virtual environment, recommended

1. (If required) Download and install [Anaconda](https://www.anaconda.com/download).
2. (If required) Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). The Microsoft C++ Build Tools are required for building the package on Windows.
3. (Recommended) Create a virtual environment to install the package into an isolated Python environment. A brief guide is provided below.
   - Start Anaconda Prompt from the Start menu.
   - Verify that `conda` is installed in your path by typing `conda -V`
   - Navigate to the `anaconda3` directory.
   - Make sure that the newest version of `conda` is installed. Update conda by typing `conda update conda`.
   - Navigate back to the previous folder.
   - Get your python version (3.x.yy) by typing `python -V`.
   - Set up a virtual environment (here named `testenv`) by typing `conda create --name testenv python=3.x` (where 3.x is replaced by the python version that you have/want to use).
   - Activate the virtual environment by typing `conda activate testenv`. To see a list of available environments, type `conda info --envs`.
   - Install [Spyder](https://www.spyder-ide.org/) into the virtual environment by typing `conda install spyder`.

### Install MASWavesPy

The package is installed using [pip](https://pip.pypa.io/en/stable/).
1. (If required) Start Anaconda Prompt.
2. Type `pip install maswavespy` to install the package.
3. Check if the package has been successfully installed by inspecting the last lines that are displayed in the Anaconda Prompt console.

### Test MASWavesPy

1. Download the contents of the [examples](https://github.com/Mazvel/maswavespy/tree/main/examples) directory (i.e., the four example `.py` files and the directory `Data`) to a folder destination of your choice.
   - The four example files (with `.py` endings) test different parts/commands of the MASWavesPy package.
   - The example files use the data from the [examples/Data](https://github.com/Mazvel/maswavespy/tree/main/examples/Data) directory as inputs. 
2. Launch the _Spyder (testenv)_ app [i.e., Spyder (name of your virtual environment)] from the Start menu.
   - _Spyder (testenv)_ is found in the folder Anaconda3 in the Start menu (for the latest versions of Anaconda).
3. Set the directory that contains the example `.py` files and the `Data` directory as the working directory in _Spyder (testenv)_.
   - The working directory is set in the top right corner of the Spyder IDE window.
4. Open and run `MASWavesPy_Dispersion_test1.py` to test the basic methods of the `wavefield` and `dispersion` modules using a single data file.
   - Please note that all four example files are written to be run one cell at a time using the keyboard shortcut (Ctrl+Enter), Run > Run cell, or the Run cell button in the toolbar.
   - Information on specific methods/commands is provided in each example file.
5. Open and run `MASWavesPy_Dispersion_test2.py` to test the methods of the `wavefield` and `dispersion` modules using a `Dataset` object.
6. Open and run `MASWavesPy_Combination_test.py` to test the `combination` module.
7. Open and run `MASWavesPy_Inversion_test.py` to test the `inversion` module.

### Deactivate the virtual environment
Applies if a virtual environment has been created.
1. (If required) Close the Spyder IDE.
2. (If required) Start Anaconda Prompt.
3. Close the virtual environment `testenv` by typing `conda deactivate`.
4. If required, the virtual environment `testenv` can be deleted with the following command `conda remove --name testenv --all`.

## Known Issues

### Matplotlib should use TkAgg on Mac

MaswavesPy depends on matplotlib. If you are on mac you need to ensure matplotlib uses `TkAgg`. Below is a workaround that is used in our examples.

```
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
```

### Tkinter not found on Mac 

On mac you might run into `ModuleNotFoundError: No module named '_tkinter'` error, even after successfully installing `maswavespy` that has [Tkinter](https://docs.python.org/3/library/tkinter.html) as one of its listed dependencies. This might be because your python3 installation did not have Tkinter correctly set up. Below is an example of how it can be installed with brew.

`brew install python-tk`

### blosc2~=2.0.0 not installed

When installing `maswavespy` into the Anaconda environment, you might encounter the following error, even though `maswavespy` is successfully installed.

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tables 3.8.0 requires blosc2~=2.0.0, which is not installed.
```

The `maswavespy` package does not require [`blosc2 2.0.0`](https://pypi.org/project/blosc2/2.0.0/). Therefore, this error message can be ignored. 

The error can be prevented by installing [`Cython`](http://www.cython.org/) (required for installing `blosc2 2.0.0`) and `blosc2 2.0.0` prior to installing  `maswavespy`. Below is an example of how these two packages can be installed

```
conda install -c conda-forge cython
pip install blosc2==2.0.0
```
