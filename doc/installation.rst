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
dependencies (requires an NVIDIA GPU and CUDA drivers)::

    pip install gplearn-CUDA[cuda]

Or if you wish to install to the home directory::

    pip install --user gplearn-CUDA

For the latest development version, first get the source from github::

    git clone https://github.com/trevorstephens/gplearn.git

Then navigate into the local ``gplearn-CUDA`` directory and simply run::

    python setup.py install

or::

    python setup.py install --user

and you're done!

Option 2: installation using conda
----------------------------------

In case you want to install ``gplearn-CUDA`` in an anaconda environment, you can run::

    conda install -c conda-forge gplearn-CUDA

and you're done!
