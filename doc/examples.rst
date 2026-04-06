.. _example:

Examples
========

The following examples demonstrate how to use ``gplearn-CUDA`` for various tasks. 
For high-performance GPU-specific examples and benchmarks, please see the 
``examples/`` directory in the source repository.

High-Performance Examples
-------------------------

Detailed examples for CUDA acceleration can be found in the repository:

*   **Regressor Tutorial**: ``examples/regressor_cuda.py``
    A step-by-step GPU example using ``SymbolicRegressor``.
*   **Classifier Tutorial**: ``examples/classifier_cuda.py``
    A GPU example using ``SymbolicClassifier``.
*   **Transformer Tutorial**: ``examples/transformer_cuda.py``
    A GPU example using ``SymbolicTransformer`` for feature generation.
*   **Benchmark Suite**: ``examples/benchmark.py``
    A comprehensive performance comparison between multi-core CPU and GPU runs 
    across various dataset scales.

Notebook Examples
-----------------

A detailed set of examples for the original ``gplearn`` functionality (all of 
which are supported by ``gplearn-CUDA``) can be
`found here <https://github.com/trevorstephens/gplearn/blob/main/doc/gp_examples.ipynb>`_
in the original repository.

.. currentmodule:: gplearn.genetic

Tutorial 1: Symbolic Regressor
------------------------------

This example is the same as the first example in the LISP GP book and seeks to
find the function :math:`y = X_0^{2} - X_1^{2} + X_1 - 1`.

.. code-block:: python

    from gplearn.genetic import SymbolicRegressor
    import numpy as np

    # Generate synthetic data
    x0 = np.arange(-1, 1, .1)
    x1 = np.arange(-1, 1, .1)
    x0, x1 = np.meshgrid(x0, x1)
    y_truth = x0**2 - x1**2 + x1 - 1

    X_train = np.vstack([x0.ravel(), x1.ravel()]).T
    y_train = y_truth.ravel()

    # Fit the model
    est = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
    est.fit(X_train, y_train)

    # Inspect the results
    print(est._program)

Output:

.. code-block:: none

    sub(add(-0.999, X1), mul(sub(X1, X0), add(X0, X1)))

The result is almost identical to the truth, with some rounding of the constant.
To run this same example on the GPU for massive speedups on larger datasets, 
simply add ``device='cuda'`` to the estimator initialization.
