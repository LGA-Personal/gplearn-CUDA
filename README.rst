.. image:: https://github.com/LGA-Personal/gplearn-CUDA/actions/workflows/build.yml/badge.svg
    :target: https://github.com/LGA-Personal/gplearn-CUDA/actions/workflows/build.yml
    :alt: Build Status
.. image:: https://img.shields.io/github/v/release/LGA-Personal/gplearn-CUDA.svg
    :target: https://github.com/LGA-Personal/gplearn-CUDA/releases
    :alt: Version
.. image:: https://img.shields.io/github/license/LGA-Personal/gplearn-CUDA.svg
    :target: https://github.com/LGA-Personal/gplearn-CUDA/blob/main/LICENSE
    :alt: License
.. image:: https://readthedocs.org/projects/gplearn-cuda/badge/?version=stable
    :target: http://gplearn-cuda.readthedocs.io/
    :alt: Documentation Status

|

Welcome to gplearn-CUDA!
========================

`gplearn-CUDA` implements GPU-accelerated Genetic Programming in Python, with a 
`scikit-learn <http://scikit-learn.org>`_ inspired and compatible API.

**This project is a high-performance extension of the original** `gplearn <https://github.com/trevorstephens/gplearn>`_ 
**library developed by Trevor Stephens.** It maintains all the original 
functionality while introducing massive parallelization via NVIDIA CUDA.

Overview
--------

While Genetic Programming (GP) can be used to perform a `very wide variety of tasks <http://www.genetic-programming.org/combined.php>`_, ``gplearn-CUDA`` focuses on three scikit-learn style estimators: symbolic regression, binary classification, and symbolic feature generation.

These map directly to ``SymbolicRegressor``, ``SymbolicClassifier``, and
``SymbolicTransformer``.

CUDA Acceleration
-----------------

The core contribution of ``gplearn-CUDA`` is high-performance **GPU acceleration** 
via NVIDIA CUDA. By setting ``device='cuda'``, the library utilizes a custom 
virtual machine interpreter on the GPU to achieve **2x–4x speedups** on massive 
datasets and large populations.

.. code-block:: python

    from gplearn.genetic import SymbolicRegressor
    # Enable CUDA acceleration
    est = SymbolicRegressor(device='cuda', population_size=5000)
    est.fit(X, y)

Installation
------------

``gplearn-CUDA`` requires a recent version of scikit-learn. The CUDA acceleration 
is compatible with **CUDA 11.2 through 13.x**.

To install the GPU-enabled version via pip (targeting CUDA 13.x by default,
including the required CUDA toolkit runtime libraries)::

    pip install gplearn-CUDA[cuda]

For other CUDA versions or Conda installation, please see the 
`Installation Guide <http://gplearn-cuda.readthedocs.io/en/stable/installation.html>`_.

Python Compatibility
--------------------

``gplearn-CUDA`` currently requires **Python 3.11 or newer** because its core
dependency ``scikit-learn>=1.8.0`` requires that Python range.

Verified in this repository on Windows:

* CPU path: **Python 3.11, 3.12, 3.13, and 3.14**
* CUDA path: **Python 3.12 and 3.14**

On some Windows Python 3.14 environments, CuPy can import and JIT kernels
successfully while optional BLAS backends such as cuBLAS still fail to load.
When that happens, ``SymbolicTransformer(device='cuda')`` now falls back to a
NumPy-based correlation step for hall-of-fame selection and emits a warning
instead of aborting the fit.

License & Credits
-----------------

``gplearn-CUDA`` is released under the **BSD 3-Clause License**, following the 
licensing of the original project.

*   Original Author: **Trevor Stephens** (`@trevorstephens <https://github.com/trevorstephens>`_)
*   CUDA Extension: **LGA-Personal**

For the latest development version, first get the source from github::

    git clone https://github.com/LGA-Personal/gplearn-CUDA.git

Then navigate into the local directory and simply run::

    pip install .

If you come across any issues in running or installing the package, `please submit a bug report <https://github.com/LGA-Personal/gplearn-CUDA/issues>`_.
