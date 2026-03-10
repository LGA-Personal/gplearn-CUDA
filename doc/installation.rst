.. _installation:

Installation
============

``gplearn-CUDA`` requires a recent version of scikit-learn (which requires numpy and
scipy). So first you will need to `follow their installation instructions <http://scikit-learn.org/dev/install.html>`_
to get the dependencies.

Option 1: installation using pip
--------------------------------

Now that you have scikit-learn installed, you can install ``gplearn-CUDA`` using pip::

    pip install gplearn-CUDA

To enable GPU/CUDA acceleration, you should also install the optional ``cuda``
dependencies. ``gplearn-CUDA`` is compatible with **CUDA 11.2 through 13.x**.

The easiest way to install the correct GPU drivers and libraries is via Conda::

    conda install -c conda-forge cupy

If you prefer using pip, you must install the specific CuPy wheel that matches 
your system's CUDA version:

- For **CUDA 13.x**: ``pip install cupy-cuda13x``
- For **CUDA 12.x**: ``pip install cupy-cuda12x``
- For **CUDA 11.x**: ``pip install cupy-cuda11x``

If you have an NVIDIA driver but do not have the CUDA Toolkit installed, you 
can install a standalone environment via pip::

    pip install "cupy-cuda12x[ctk]"

Or if you wish to install to the home directory::


    pip install --user gplearn-CUDA

For the latest development version, first get the source from github::

    git clone https://github.com/LGA-Personal/gplearn-CUDA.git

Then navigate into the local ``gplearn-CUDA`` directory and simply run::

    pip install .

or::

    pip install --user .

and you're done!

Option 2: installation using conda
----------------------------------

In case you want to install ``gplearn-CUDA`` in an anaconda environment, you can run::

    conda install -c conda-forge gplearn-CUDA

and you're done!
