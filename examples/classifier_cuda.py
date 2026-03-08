"""
SymbolicClassifier CUDA Tutorial
================================

This example demonstrates how to use the SymbolicClassifier with GPU acceleration 
to solve a binary classification problem.
"""

import numpy as np
import time
from gplearn.genetic import SymbolicClassifier
from sklearn.metrics import roc_auc_score

def run_classifier_tutorial():
    print("--- gplearn-CUDA: SymbolicClassifier Tutorial ---")
    
    # Generate 200k samples with 10 features
    n_samples = 200_000
    n_features = 10
    print(f"Generating data: {n_samples} samples, {n_features} features...")
    
    X = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    # Target: 1 if x0^2 + x1 > 0.5, else 0
    y = (X[:, 0]**2 + X[:, 1] > 0.5).astype(int)

    # Initialize Classifier with device='cuda'
    # Note: Log-loss is optimized on the GPU
    clf = SymbolicClassifier(
        population_size=2000,
        generations=10,
        tournament_size=20,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        verbose=1,
        random_state=42,
        device='cuda'  # Enable high-speed GPU path
    )

    print("\nFitting classifier on GPU...")
    start = time.time()
    clf.fit(X, y)
    print(f"Fit Time: {time.time() - start:.2f}s")

    # Predict probabilities using GPU-accelerated sigmoid path
    print("\nPredicting probabilities...")
    y_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_proba[:, 1])
    print(f"ROC AUC: {auc:.6f}")
    print(f"Best Program: {clf._program}")

if __name__ == "__main__":
    run_classifier_tutorial()
