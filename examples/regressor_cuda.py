"""
SymbolicRegressor CUDA Tutorial
===============================

This example demonstrates how to use the SymbolicRegressor with GPU acceleration 
on a large-scale regression task.
"""

import numpy as np
import time
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error

def run_regressor_tutorial():
    print("--- gplearn-CUDA: SymbolicRegressor Tutorial ---")
    
    # Generate 500k samples with 10 features
    n_samples = 500_000
    n_features = 10
    print(f"Generating data: {n_samples} samples, {n_features} features...")
    
    X = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    # y = x0^2 + x1*x2 - sin(x3) + 0.5
    y = X[:, 0]**2 + X[:, 1]*X[:, 2] - np.sin(X[:, 3]) + 0.5
    y = y.astype(np.float32)

    # Initialize Regressor with device='cuda'
    est = SymbolicRegressor(
        population_size=5000,
        generations=20,
        tournament_size=20,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=1,
        random_state=42,
        device='cuda'  # Enable high-speed GPU path
    )

    print("\nFitting model on GPU...")
    start = time.time()
    est.fit(X, y)
    print(f"Fit Time: {time.time() - start:.2f}s")

    # Predict using the JIT-compiled GPU path
    print("\nPredicting on training data...")
    y_pred = est.predict(X)
    print(f"MSE: {mean_squared_error(y, y_pred):.6f}")
    print(f"Best Program: {est._program}")

if __name__ == "__main__":
    run_regressor_tutorial()
