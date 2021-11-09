# Python Algorithms for Randomized Linear Algebra (PARLA)

PARLA is a Python package for prototyping the *mathematical* structure of a future C++ library for randomized numerical linear algebra.
The future library is meant to be "LAPACK-like" and will organize its functionality into high-level "drivers" and lower-level "computational routines".

The main difference between PARLA and LAPACK (aside from LAPACK being written in Fortran!) is that PARLA defines
algorithms in an object-oriented way.
This approach makes it easy to modify a driver's implementation by overriding the default implementation of some constituent computational routine.
All driver-level functionality has passed basic tests.
We're actively developing the testing framework and improving documentation.

PARLA has a companion Matlab library called [MARLA](https://github.com/BallisticLA/marla).
MARLA has a purely procedural design and isn't meant to be as general as PARLA.
We're in the process of standardizing a procedural API that's consistent across these two libraries.
The state of PARLA and MARLA's APIs and unit tests is summarized in [this Google Sheets spreadsheet](https://docs.google.com/spreadsheets/d/15vIS5wkaVB5lUoVQZqg7J_2qsK04ycVN47Mo2LuIKAo/edit?usp=sharing).
(It will be hard to read that spreadsheet without having the RandLAPACK design document on-hand.)

There are no plans to distribute this Python package through PyPI or conda.

## How to install
The following detailed instructions assume you have Git available from the command line.
You might need to modify the commands if you're running Windows or macOS.

  1. Make sure you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) installed.
  2. Move to a directory where you can make a folder for this repo.
       * If you just want to try out notebooks, that directory might be something like ``~/Desktop/temp/``.
  3. Get the source code by running ``git clone https://github.com/rileyjmurray/parla.git``.
       * That will create a folder like ``~/Desktop/temp/parla`` that contains the contents of this repo.
       * Change directory so you're in the new folder. You should have ``setup.py`` in your working directory.
  4. Create and activate a new python environment.
       * Decide on an environment name.
          * For concreteness, I'll use the name ``rla39a`` moving forward.
       * If you have an Intel machine and want to link into MKL, do the following.
           * Run ``conda create --name rla39a python=3.9 pytest mkl -y``,
           * Run ``conda activate rla39a``,
           * Run ``conda install -c intel numpy -y``,
           * Run ``conda install scipy jupyter matplotlib -y``.
       * If you don't have an Intel machine or if you want to use OpenBLAS, do the following.
           * Run ``conda create --name rla39a python=3.9 pytest numpy scipy jupyter matplotlib -y``.
           * Run ``conda activate rla39a``.
  5. Install parla by running ``pip install -e .``.
       * You need to be in the same directory that contains parla's ``setup.py`` file.
       * This command makes it possible to import parla from python, no matter your working directory.
       * The ``-e`` flag means that any edits to parla source code will be incorporated on future imports.
  6. Optional: run unittests with the command  ``pytest parla``
       * You need to be in the same directory that contains parla's ``setup.py`` file.
  7. Optional: verify that NumPy and SciPy are linked against the expected BLAS and LAPACK implementations
       * Run `` python -c "import numpy as np; np.show_config()"``
       * The command above will probably mention MKL or OpenBLAS. If you wanted MKL and it makes *any*
         mention of OpeBLAS, then something went wrong in the installation process.
         Email me (rjmurray@berkeley.edu) for help.

### Notes on MKL vs OpenBLAS

Unless you go out of your way to install a version of NumPy that's linked to MKL, you'll almost certainly
end up with NumPy and SciPy getting linked against OpenBLAS. OpenBLAS comes with an LAPACK implementation, however,
it's almost a direct copy from the Netlib LAPACK implementation. That implementation is very inefficient
(all things considered) and results in unnecessarily poor performance for some Python  functions that call
into LAPACK (particularly, SciPy's least squares solver). I strongly recommend that you use MKL if possible.

## How to uninstall
The installation process above might take up a nontrivial amount of space on your computer. For example, the Intel MKL library is around 200 megabytes. You might want to delete the python environement if you're certain that you're done working with parla. If you named your environment ``rla39a`` like above, then you'd run ``conda env remove --name rla39a``. Make sure you are in a *different* python environment before running that command.
 
## How to run Jupyter Notebooks
The following instructions are very generic.
However, they assume you've gone through the installation process described above.

  1. Make sure to activate the python environment with your parla installation.
      * From the installation example, the command would be ``conda activate rla39a``.
  2. Move to any directory on your computer that has your desired notebook somewhere in its subdirectories.
      * You don't need to have the notebook in your working directory.
  3. Run ``jupyter-notebook``.
      * This should print out some messages and might launch a browser window.
      * The messages printed by this command should include two or three URLs; go to the last URL.

Your browser should now be running a Jupyter Notebook server. You can navigate to the notebook you want to run (such as [one of](https://github.com/rileyjmurray/parla/blob/main/notebooks/least_squares/procedural_least_squares_driver.ipynb) [these two](https://github.com/rileyjmurray/parla/blob/main/notebooks/least_squares/sap1_vs_lapack.ipynb)) and launch it.
