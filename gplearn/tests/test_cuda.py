import numpy as np
import pytest
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier, SymbolicTransformer
from gplearn._program import _Program, _batch_evaluate_gpu
from gplearn.utils import HAS_CUDA

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_numerical_parity():
    """Verify that GPU VM execution matches CPU stack-based execution."""
    X = np.random.uniform(-1, 1, (1000, 5)).astype(np.float32)
    y = X[:, 0]**2 + X[:, 1]
    
    # Regression
    est_cpu = SymbolicRegressor(population_size=100, generations=1, random_state=42, device='cpu')
    est_gpu = SymbolicRegressor(population_size=100, generations=1, random_state=42, device='cuda')
    
    est_cpu.fit(X, y)
    est_gpu.fit(X, y)
    
    cpu_pred = est_cpu.predict(X)
    gpu_pred = est_gpu.predict(X)
    
    # Check max absolute difference
    np.testing.assert_allclose(cpu_pred, gpu_pred, atol=1e-5)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_batch_evaluation():
    """Verify that batch evaluation returns same results as individual execution."""
    import cupy as cp
    X = np.random.uniform(-1, 1, (500, 10)).astype(np.float32)
    X_gpu = cp.asarray(X).T.copy()
    
    # Create multiple programs
    est = SymbolicRegressor(population_size=50, generations=1, random_state=42)
    est.fit(X, X[:, 0]) # dummy fit to get programs
    
    population = est._programs[0]
    
    # Batch evaluate
    y_batch = _batch_evaluate_gpu(population, X_gpu, X.shape[0], X.shape[1])

    # Compare with individual execute()
    for i, program in enumerate(population):
        y_ind = np.asarray(program.execute(X), dtype=np.float32)
        np.testing.assert_allclose(y_batch[i].get(), y_ind,
                                   rtol=5e-5, atol=2e-2)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_vm_stack_limits():
    """Verify the VM handles deep programs without crashing (within stack limits)."""
    from gplearn.functions import add2

    # Prefix form for a 51-leaf left-deep addition tree.
    program = [add2] * 50 + [0] * 51
    prog = _Program(function_set=[add2], arities={2: [add2]}, init_depth=(2, 2),
                    init_method='full', n_features=1, const_range=None,
                    metric=None, p_point_replace=0.1, parsimony_coefficient=0.1,
                    random_state=np.random.RandomState(42), device='cuda',
                    program=program)

    X = np.random.uniform(-1, 1, (128, 1)).astype(np.float32)
    result = prog.execute(X).get()
    np.testing.assert_allclose(result, 51 * X[:, 0], rtol=1e-6, atol=5e-5)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_transposition_heuristics():
    """Verify that execute() correctly identifies and transposes input data."""
    X = np.random.uniform(-1, 1, (100, 5)).astype(np.float32)
    est = SymbolicRegressor(device='cuda')
    est.n_features_in_ = 5
    
    # Mock a program
    from gplearn.functions import add2
    prog = _Program(function_set=[add2], arities={2: [add2]}, init_depth=(2, 2),
                    init_method='full', n_features=5, const_range=(0, 1),
                    metric=None, p_point_replace=0.1, parsimony_coefficient=0.1,
                    random_state=np.random.RandomState(42), device='cuda',
                    program=[add2, 0, 1])
    
    # Pass (n_samples, n_features) - should be auto-transposed
    res = prog.execute(X)
    assert res.shape == (100,)
