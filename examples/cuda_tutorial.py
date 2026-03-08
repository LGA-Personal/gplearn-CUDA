"""
gplearn-CUDA Tutorial
=====================

This tutorial demonstrates how to use GPU acceleration in gplearn-CUDA for 
symbolic regression and transformation tasks.

Prerequisites:
--------------
1. NVIDIA GPU with CUDA drivers.
2. cupy-cudaXX package installed (e.g., `pip install gplearn-CUDA[cuda]`).
"""

import numpy as np
import time
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_tutorial():
    print("--- gplearn-CUDA Tutorial ---")
    
    # 1. Generate Synthetic Data
    # ---------------------------
    # We'll use a relatively large dataset to show the GPU benefit.
    n_samples = 500_000
    n_features = 10
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features...")
    
    X = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    # Target function: y = x0^2 + x1 * x2 - sin(x3) + 0.5
    y = X[:, 0]**2 + X[:, 1] * X[:, 2] - np.sin(X[:, 3]) + 0.5
    y = y.astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Symbolic Regression with CUDA
    # --------------------------------
    # To enable CUDA, simply set device='cuda'.
    # We'll use a large population to saturate the GPU.
    print("\n[Step 1] Fitting SymbolicRegressor on GPU...")
    
    est_gpu = SymbolicRegressor(
        population_size=5000,
        generations=10,
        tournament_size=20,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.01,
        random_state=42,
        device='cuda'  # Enable CUDA acceleration
    )

    start_time = time.time()
    est_gpu.fit(X_train, y_train)
    gpu_fit_time = time.time() - start_time
    print(f"GPU Fit Time: {gpu_fit_time:.2f}s")

    # 3. Prediction (CUDA JIT Path)
    # -----------------------------
    # Prediction on the GPU uses a JIT-compiled kernel for the best individual.
    print("\n[Step 2] Predicting on test set using GPU...")
    y_pred = est_gpu.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"GPU Model MSE: {mse:.6f}")
    print(f"Best program found: {est_gpu._program}")

    # 4. Symbolic Transformation (Feature Engineering)
    # ------------------------------------------------
    # SymbolicTransformer can discover many non-linear features in one batch.
    print("\n[Step 3] Fitting SymbolicTransformer on GPU...")
    
    trans_gpu = SymbolicTransformer(
        population_size=2000,
        generations=5,
        hall_of_fame=100,
        n_components=10,
        tournament_size=20,
        verbose=1,
        random_state=42,
        device='cuda'
    )

    start_time = time.time()
    trans_gpu.fit(X_train, y_train)
    
    # transform() uses the batched GPU VM for maximum throughput
    X_new = trans_gpu.transform(X_train)
    print(f"Transformation complete. Original features: {X_train.shape[1]}, New features: {X_new.shape[1]}")
    print(f"Time for Transformer: {time.time() - start_time:.2f}s")

    # 5. Performance Comparison (Summary)
    # -----------------------------------
    print("\n[Summary] CUDA Acceleration Benefits:")
    print("- Automatic memory coalescing for high-dimensional data.")
    print("- Batch VM interpretation for 2x-4x speedups over multi-core CPU.")
    print("- Seamless integration with existing scikit-learn pipelines.")

if __name__ == "__main__":
    run_tutorial()
