# RLApy
This is a python library for prototyping the *mathematical* structure of a (future) C++ library for randomized linear algebra.
There are no plans to distribute this python library through PyPI or conda.

## How to install
The following instructions assume you have Git available from the command line.
You might need to modify the commands if you're running Windows or macOS.

  1. Make sure you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) installed.
  2. Move to a directory where you can make a folder for this repo.
       * If you just want to try out notebooks, that directory might be something like ``~/Desktop/temp/``.
  3. Get the source code by running ``git clone https://github.com/rileyjmurray/rlapy.git``.
       * That will create a folder like ``~/Desktop/temp/rlapy`` that contains the contents of this repo.
       * Change directory so you're in the new folder. You should have ``setup.py`` in your working directory.
  4. Create and activate a new python environment.
       * Decide on an environment name.
          * For concreteness, I'll use the name ``rla39a`` moving forward.
       * If you have an Intel machine and want to link into MKL, do the following.
           * Run ``conda create --name rla39a python=3.9 pytest mkl -y``.
           * Run ``conda install numpy scipy jupyter matplotlib -y``.
       * If you don't have an Intel machine or if you want to use OpenBLAS, do the following.
           * Run ``conda create --name rla39a python=3.9 pytest numpy scipy jupyter matplotlib -y``.
       * Run ``conda activate rla39a``.
  5. Install rlapy by running ``pip install -e .``.
       * You need to be in the same directory that contains rlapy's ``setup.py`` file.
       * This command makes it possible to import rlapy from python, no matter your working directory.
       * The ``-e`` flag means that any edits to rlapy source code will be incorporated on future imports.
  6. Optional: run unittests with the command  ``pytest rlapy``
       * You need to be in the same directory that contains rlapy's ``setup.py`` file.

## How to uninstall
The installation process above might take up a nontrivial amount of space on your computer. For example, the Intel MKL library is around 200 megabytes. You might want to delete the python environement if you're certain that you're done working with rlapy. If you named your environment ``rla39a`` like above, then you'd run ``conda env remove --name rla39a``. Make sure you are in a *different* python environment before running that command.
 
## How to run Jupyter Notebooks
The following instructions are very generic.
However, they assume you've gone through the installation process described above.

  1. Make sure to activate the python environment with your rlapy installation.
      * From the installation example, the command would be ``conda activate rla39a``.
  2. Move to any directory on your computer that has your desired notebook somewhere in its subdirectories.
      * You don't need to have the notebook in your working directory.
  3. Run ``jupyter-notebook``.
      * This should print out some messages and might launch a browser window.
      * The messages printed by this command should include two or three URLs; go to the last URL.

Your browser should now be running a Jupyter Notebook server. You can navigate to the notebook you want to run (such as [one of](https://github.com/rileyjmurray/rlapy/blob/main/notebooks/least_squares/procedural_least_squares_driver.ipynb) [these two](https://github.com/rileyjmurray/rlapy/blob/main/notebooks/least_squares/sap1_vs_lapack.ipynb)) and launch it.
