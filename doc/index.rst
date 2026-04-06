.. gplearn documentation master file, created by
   sphinx-quickstart on Sun Apr 19 18:40:35 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gplearn-CUDA's documentation!
========================================

|

.. image:: logos/gplearn-wide.png
    :align: center

|

.. math::
    One \,general \,law, \,leading \,to \,the \,advancement \,of \,all \,organic \,beings, namely,

.. math::
    multiply, \,vary, \,let \,the \,strongest \,live \,and \,the \,weakest \,die.

.. math::    
    - Charles \,Darwin, \,On \,the \,Origin \,of \,Species \,(1859)

|
|

.. currentmodule:: gplearn.genetic

``gplearn-CUDA`` implements GPU-accelerated Genetic Programming in Python, with 
a `scikit-learn <http://scikit-learn.org>`_ inspired and compatible API.

**This project is a high-performance extension of the original** `gplearn <https://github.com/trevorstephens/gplearn>`_ 
**library developed by Trevor Stephens.** It maintains all the original 
functionality while introducing massive parallelization via NVIDIA CUDA.

``gplearn-CUDA`` supports regression through the :class:`SymbolicRegressor`,
binary classification with the :class:`SymbolicClassifier`, and automated
feature engineering with the :class:`SymbolicTransformer`. All three share the
same GP evolution engine while keeping the familiar scikit-learn estimator
workflow.

``gplearn-CUDA`` retains the familiar scikit-learn ``fit``/``predict`` API and
works with the existing scikit-learn `pipeline <https://scikit-learn.org/stable/modules/compose.html>`_
and `grid search <http://scikit-learn.org/stable/modules/grid_search.html>`_
modules. You can get started with ``gplearn-CUDA`` as simply as::

    est = SymbolicRegressor()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)

However, don't let that stop you from exploring all the ways that the evolution
can be tailored to your problem. The package attempts to squeeze a lot of
functionality into a scikit-learn-style API. While there are a lot of
parameters to tweak, reading the documentation here should make the more
relevant ones clear for your problem.

``gplearn-CUDA`` is built on scikit-learn and a fairly recent copy is required
for installation. If you come across any issues in running or installing the
package, `please submit a bug report <https://github.com/LGA-Personal/gplearn-CUDA/issues>`_.

Next up, read some more details about :ref:`what Genetic Programming is <intro>`,
and how it works...

Contents:

.. toctree::
   :maxdepth: 2

   intro
   examples
   reference
   cuda_architecture
   advanced
   installation
   contributing
   changelog
