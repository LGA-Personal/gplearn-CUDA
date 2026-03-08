"""
SymbolicTransformer CUDA Tutorial
=================================

This example demonstrates how to use the SymbolicTransformer with GPU acceleration 
to discover non-linear features in a dataset.
"""

import numpy as np
import time
from gplearn.genetic import SymbolicTransformer

def run_transformer_tutorial():
    print("--- gplearn-CUDA: SymbolicTransformer Tutorial ---")
    
    # Generate 300k samples with 10 features
    n_samples = 300_000
    n_features = 10
    print(f"Generating data: {n_samples} samples, {n_features} features...")
    
    X = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    # Target is a hidden interaction of a few features
    y = X[:, 0]**2 + X[:, 1]*X[:, 2] - X[:, 3]
    y = y.astype(np.float32)

    # Initialize Transformer with device='cuda'
    # hall_of_fame: top programs to consider for the final feature set
    # n_components: number of final, non-correlated features to output
    trans = SymbolicTransformer(
        population_size=2000,
        generations=10,
        hall_of_fame=100,
        n_components=5,
        verbose=1,
        random_state=42,
        device='cuda'  # Enable high-speed GPU path
    )

    print("\nFitting transformer on GPU...")
    start = time.time()
    trans.fit(X, y)
    print(f"Fit Time: {time.time() - start:.2f}s")

    # transform() uses the batched GPU VM for maximum throughput
    print("\nTransforming dataset to new feature space...")
    start_trans = time.time()
    X_new = trans.transform(X)
    print(f"Transformation Time: {time.time() - start_trans:.2f}s")
    print(f"Original shape: {X.shape}, Transformed shape: {X_new.shape}")
    
    print("\nDiscovered Features (Top 5):")
    for i, program in enumerate(trans):
        print(f"Feature {i}: {program}")

if __name__ == "__main__":
    run_transformer_tutorial()
