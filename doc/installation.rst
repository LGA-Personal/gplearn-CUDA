.. _installation:

Installation
============

``gplearn-CUDA`` requires a recent version of scikit-learn (which requires numpy and
scipy). So first you will need to `follow their installation instructions <http://scikit-learn.org/dev/install.html>`_
to get the dependencies.

Python support currently begins at **Python 3.11**, matching the requirement
imposed by ``scikit-learn>=1.8.0``.

Option 1: installation using pip
--------------------------------

Now that you have scikit-learn installed, you can install ``gplearn-CUDA`` using pip::

    pip install gplearn-CUDA

To enable GPU/CUDA acceleration, you should also install the optional ``cuda``
dependencies. ``gplearn-CUDA`` is compatible with **CUDA 11.2 through 13.x**.
The package extra currently installs the CuPy build for **CUDA 13.x** by
default::

    pip install "gplearn-CUDA[cuda]"

The easiest way to install the correct GPU drivers and libraries is via Conda::

    conda install -c conda-forge cupy

If you prefer using pip, you must install the specific CuPy wheel that matches
your system's CUDA version, then install ``gplearn-CUDA`` itself:

- For **CUDA 13.x**: ``pip install cupy-cuda13x``
- For **CUDA 12.x**: ``pip install cupy-cuda12x``
- For **CUDA 11.x**: ``pip install cupy-cuda11x``

Then::

    pip install gplearn-CUDA

If you have an NVIDIA driver but do not have the CUDA Toolkit installed, you 
can install a standalone environment via pip::

    pip install "cupy-cuda13x[ctk]"

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

Notes
-----

- Verified CPU support in this repository covers Python 3.11, 3.12, 3.13 and
  3.14 on Windows.
- Verified CUDA support in this repository covers Python 3.12 and 3.14 on
  Windows.
- On some Windows Python 3.14 environments, CuPy can import and compile CUDA
  kernels while optional BLAS backends such as cuBLAS still fail to load. In
  that case, ``SymbolicTransformer(device='cuda')`` falls back to a NumPy
  correlation step and emits a warning instead of aborting.
