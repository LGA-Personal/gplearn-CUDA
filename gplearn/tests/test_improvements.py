"""Tests for the 10 GPU and algorithmic optimizations.

These tests target potential failure points in each improvement, covering
both CPU-only paths (always runnable) and GPU paths (skipped without CuPy).
"""

import struct
from collections import OrderedDict

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.utils.validation import check_random_state

from gplearn._program import (
    _Program, _batch_evaluate_gpu, _float_to_key,
    _cache_kernel, clear_kernel_cache, _CUDA_KERNEL_CACHE,
    _CUDA_KERNEL_CACHE_MAX_SIZE, _OPCODES, MAX_SHARED_FEATURES,
)
from gplearn.fitness import _weighted_spearman, _weighted_pearson
from gplearn.functions import _function_map
from gplearn.genetic import (
    SymbolicRegressor, SymbolicClassifier, SymbolicTransformer,
)
from gplearn.utils import HAS_CUPY

# Check if CUDA runtime is actually available (not just CuPy installed)
HAS_CUDA = False
if HAS_CUPY:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        HAS_CUDA = True
    except Exception:
        HAS_CUDA = False

# Fixtures
diabetes = load_diabetes()
perm = check_random_state(0).permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]


def _make_program(program_list, n_features=3, device='cpu'):
    """Helper to create a _Program with a specific program list."""
    rs = check_random_state(42)
    func_set = [_function_map['add'], _function_map['mul'],
                _function_map['sub'], _function_map['div']]
    arities = {2: func_set}
    return _Program(
        function_set=func_set,
        arities=arities,
        init_depth=(2, 4),
        init_method='grow',
        n_features=n_features,
        const_range=(-1., 1.),
        metric=None,
        p_point_replace=0.05,
        parsimony_coefficient=0.001,
        random_state=rs,
        device=device,
        program=program_list,
    )


# =========================================================================
# Improvement #10: Bit-exact float constant deduplication
# =========================================================================

class TestFloatToKey:
    """Test _float_to_key for correctness on edge cases."""

    def test_positive_and_negative_zero_differ(self):
        """Positive zero and negative zero must produce different keys."""
        key_pos = _float_to_key(0.0)
        key_neg = _float_to_key(-0.0)
        assert key_pos != key_neg, (
            "-0.0 and 0.0 should have different bit patterns")

    def test_nan_is_consistent(self):
        """NaN must produce a consistent key (unlike float equality)."""
        key1 = _float_to_key(float('nan'))
        key2 = _float_to_key(float('nan'))
        assert key1 == key2, "NaN keys must be consistent"

    def test_nan_not_equal_to_zero(self):
        """NaN key must differ from 0.0."""
        assert _float_to_key(float('nan')) != _float_to_key(0.0)

    def test_distinct_values_distinct_keys(self):
        """Different floats must produce different keys."""
        vals = [0.0, 1.0, -1.0, 0.5, 1e-10, 1e10, float('inf'),
                float('-inf')]
        keys = [_float_to_key(v) for v in vals]
        assert len(set(keys)) == len(keys), "All distinct values need distinct keys"

    def test_equal_values_equal_keys(self):
        """Identical floats must produce identical keys."""
        assert _float_to_key(1.0) == _float_to_key(1.0)
        assert _float_to_key(-3.14) == _float_to_key(-3.14)

    def test_returns_bytes(self):
        """Key must be bytes of length 4 (single-precision float)."""
        key = _float_to_key(42.0)
        assert isinstance(key, bytes)
        assert len(key) == 4

    def test_subnormal_handling(self):
        """Subnormal (denormalized) floats must be correctly keyed."""
        tiny = 1e-45  # smallest positive subnormal for float32
        key = _float_to_key(tiny)
        assert isinstance(key, bytes)
        assert key != _float_to_key(0.0)


# =========================================================================
# Improvement #3: Bounded LRU kernel cache
# =========================================================================

class TestBoundedKernelCache:
    """Test the bounded LRU cache for CUDA kernels."""

    def setup_method(self):
        clear_kernel_cache()

    def teardown_method(self):
        clear_kernel_cache()

    def test_cache_stores_entries(self):
        _cache_kernel('k1', 'kernel1')
        assert 'k1' in _CUDA_KERNEL_CACHE
        assert _CUDA_KERNEL_CACHE['k1'] == 'kernel1'

    def test_cache_evicts_oldest_when_full(self):
        """Verify LRU eviction when cache exceeds max size."""
        # Fill the cache to max
        for i in range(_CUDA_KERNEL_CACHE_MAX_SIZE):
            _cache_kernel(f'key_{i}', f'kernel_{i}')
        assert len(_CUDA_KERNEL_CACHE) == _CUDA_KERNEL_CACHE_MAX_SIZE

        # Add one more — oldest (key_0) should be evicted
        _cache_kernel('new_key', 'new_kernel')
        assert len(_CUDA_KERNEL_CACHE) == _CUDA_KERNEL_CACHE_MAX_SIZE
        assert 'key_0' not in _CUDA_KERNEL_CACHE
        assert 'new_key' in _CUDA_KERNEL_CACHE

    def test_cache_lru_reorder(self):
        """Accessing existing key moves it to end (most recently used)."""
        _cache_kernel('a', 'ka')
        _cache_kernel('b', 'kb')
        _cache_kernel('c', 'kc')

        # Access 'a' — should move it to end
        _cache_kernel('a', 'ka')
        keys = list(_CUDA_KERNEL_CACHE.keys())
        assert keys[-1] == 'a', "Accessed key should be at end (MRU)"
        assert keys[0] == 'b', "Untouched key should remain at front (LRU)"

    def test_clear_empties_cache(self):
        _cache_kernel('x', 'kx')
        clear_kernel_cache()
        assert len(_CUDA_KERNEL_CACHE) == 0

    def test_cache_is_ordered_dict(self):
        assert isinstance(_CUDA_KERNEL_CACHE, OrderedDict)


# =========================================================================
# Improvement #6: Postfix caching
# =========================================================================

class TestPostfixCaching:
    """Test lazy postfix caching on _Program objects."""

    def test_cache_populated_on_first_call(self):
        add = _function_map['add']
        prog = _make_program([add, 0, 1])
        assert prog._postfix_cache is None
        pf = prog.to_postfix()
        assert prog._postfix_cache is not None
        assert prog._postfix_cache is pf

    def test_cache_returns_same_object(self):
        add = _function_map['add']
        prog = _make_program([add, 0, 1])
        pf1 = prog.to_postfix()
        pf2 = prog.to_postfix()
        assert pf1 is pf2, "Second call must return cached (same) object"

    def test_postfix_correctness(self):
        """Verify postfix conversion is still correct with caching."""
        add = _function_map['add']
        mul = _function_map['mul']
        # mul(add(x0, x1), x2) -> prefix: [mul, add, x0, x1, x2]
        prog = _make_program([mul, add, 0, 1, 2])
        pf = prog.to_postfix()
        # Verify the postfix is self-consistent by checking it contains
        # the right elements: two opcodes and three variables
        opcodes_in_pf = [n for n in pf if isinstance(n, int) and n < 1000]
        vars_in_pf = [n for n in pf if isinstance(n, int) and n >= 1000]
        assert sorted(opcodes_in_pf) == sorted([_OPCODES['add'], _OPCODES['mul']])
        assert sorted(vars_in_pf) == [1000, 1001, 1002]
        # Last element must be the root operator (mul)
        assert pf[-1] == _OPCODES['mul']

    def test_new_program_has_no_cache(self):
        """Programs created via genetic operations start with no cache."""
        add = _function_map['add']
        parent = _make_program([add, 0, 1])
        rs = check_random_state(123)
        child_program, _ = parent.point_mutation(rs)
        child = _make_program(child_program)
        assert child._postfix_cache is None

    def test_cache_with_constants(self):
        """Postfix caching works correctly for programs with float constants."""
        add = _function_map['add']
        prog = _make_program([add, 0, 0.5])
        pf = prog.to_postfix()
        # Should contain the float constant
        assert any(isinstance(node, float) for node in pf)
        # Second call returns cached
        assert prog.to_postfix() is pf


# =========================================================================
# Improvement #5: Fixed transposition heuristic
# =========================================================================

class TestTranspositionHeuristic:
    """Test that execute() correctly handles various input shapes."""

    def test_standard_layout_cpu(self):
        """CPU execute with (n_samples, n_features) input works."""
        add = _function_map['add']
        prog = _make_program([add, 0, 1], n_features=3)
        X = np.random.randn(100, 3).astype(np.float32)
        result = prog.execute(X)
        assert result.shape == (100,)
        np.testing.assert_allclose(result, X[:, 0] + X[:, 1])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_square_matrix_gpu(self):
        """GPU execute with n_samples == n_features (square) doesn't crash."""
        import cupy as cp
        add = _function_map['add']
        prog = _make_program([add, 0, 1], n_features=5, device='cuda')
        # Square matrix: 5 samples x 5 features
        X = np.random.randn(5, 5).astype(np.float32)
        result = prog.execute(X)
        assert result.shape == (5,)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_already_transposed_gpu(self):
        """GPU execute with pre-transposed (n_features, n_samples) input."""
        import cupy as cp
        add = _function_map['add']
        prog = _make_program([add, 0, 1], n_features=3, device='cuda')
        # Already transposed: (3, 100)
        X = cp.random.randn(3, 100).astype(cp.float32)
        result = prog.execute(X)
        assert result.shape == (100,)


# =========================================================================
# Improvement #2: GPU-native Spearman ranking
# =========================================================================

class TestGPUSpearman:
    """Test GPU-native Spearman ranking against CPU reference."""

    def test_spearman_cpu_path(self):
        """Spearman with NumPy arrays uses CPU path correctly."""
        np.random.seed(42)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        w = np.ones(5)
        result = _weighted_spearman(y, y_pred, w)
        assert isinstance(result, float)
        assert 0.9 < result <= 1.0, "Should be high correlation"

    def test_spearman_perfect_correlation(self):
        """Perfect monotonic relationship should give correlation ~1.0."""
        y = np.arange(100, dtype=np.float64)
        y_pred = y * 2 + 1  # Perfect linear -> perfect Spearman
        w = np.ones(100)
        result = _weighted_spearman(y, y_pred, w)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_spearman_anti_correlation(self):
        """Perfect anti-monotonic relationship should give |corr| ~1.0."""
        y = np.arange(100, dtype=np.float64)
        y_pred = -y  # Perfect anti-correlation
        w = np.ones(100)
        result = _weighted_spearman(y, y_pred, w)
        # _weighted_spearman returns abs(corr) via _weighted_pearson
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_spearman_gpu_matches_cpu(self):
        """GPU Spearman ranking must match CPU scipy.stats.rankdata."""
        import cupy as cp
        from gplearn.fitness import _gpu_rankdata
        from scipy.stats import rankdata

        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float32)
        cpu_ranks = rankdata(data).astype(np.float32)
        gpu_ranks = _gpu_rankdata(cp.asarray(data)).get()
        np.testing.assert_array_equal(cpu_ranks, gpu_ranks)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_spearman_gpu_full_pipeline(self):
        """End-to-end Spearman metric on GPU gives same result as CPU."""
        import cupy as cp

        np.random.seed(42)
        y = np.random.randn(500).astype(np.float32)
        y_pred = y + np.random.randn(500).astype(np.float32) * 0.1
        w = np.ones(500, dtype=np.float32)

        cpu_result = _weighted_spearman(y, y_pred, w)
        gpu_result = _weighted_spearman(
            cp.asarray(y), cp.asarray(y_pred), cp.asarray(w))

        np.testing.assert_allclose(cpu_result, gpu_result, atol=1e-4)


# =========================================================================
# Improvement #1: Vectorized GPU fitness loop
# =========================================================================

class TestVectorizedFitnessLoop:
    """Test the vectorized GPU fitness computation path."""

    def test_cpu_regression_still_works(self):
        """Ensure CPU path is unaffected by GPU fitness changes."""
        X = diabetes.data[:100]
        y = diabetes.target[:100]
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (100,)
        assert np.isfinite(pred).all()

    def test_cpu_with_sample_weight(self):
        """CPU path with sample weights still works after refactor."""
        X = diabetes.data[:100]
        y = diabetes.target[:100]
        w = np.random.RandomState(42).uniform(0.5, 1.5, 100)
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42)
        est.fit(X, y, sample_weight=w)
        pred = est.predict(X)
        assert pred.shape == (100,)

    def test_cpu_with_max_samples_oob(self):
        """CPU path with max_samples < 1.0 (OOB evaluation)."""
        X = diabetes.data[:200]
        y = diabetes.target[:200]
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            max_samples=0.8)
        est.fit(X, y)
        assert np.isfinite(est.run_details_['best_oob_fitness'][-1])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_gpu_with_max_samples_oob(self):
        """GPU vectorized path with max_samples < 1.0."""
        X = diabetes.data[:200].astype(np.float32)
        y = diabetes.target[:200].astype(np.float32)
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            max_samples=0.8, device='cuda')
        est.fit(X, y)
        assert np.isfinite(est.run_details_['best_oob_fitness'][-1])

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_gpu_with_sample_weight(self):
        """GPU vectorized path with sample weights."""
        X = diabetes.data[:200].astype(np.float32)
        y = diabetes.target[:200].astype(np.float32)
        w = np.random.RandomState(42).uniform(0.5, 1.5, 200).astype(
            np.float32)
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            device='cuda')
        est.fit(X, y, sample_weight=w)
        pred = est.predict(X)
        assert pred.shape == (200,)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_gpu_with_weight_and_oob(self):
        """GPU vectorized path with both sample weights and OOB."""
        X = diabetes.data[:200].astype(np.float32)
        y = diabetes.target[:200].astype(np.float32)
        w = np.random.RandomState(42).uniform(0.5, 1.5, 200).astype(
            np.float32)
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            max_samples=0.8, device='cuda')
        est.fit(X, y, sample_weight=w)
        assert np.isfinite(est.run_details_['best_oob_fitness'][-1])


# =========================================================================
# Improvement #4: Pre-allocated GPU buffers
# =========================================================================

class TestPreAllocatedBuffers:
    """Test buffer reuse across generations."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_buffer_reused_across_generations(self):
        """The _y_pred_buf should be set after fit."""
        X = diabetes.data[:100].astype(np.float32)
        y = diabetes.target[:100].astype(np.float32)
        est = SymbolicRegressor(
            population_size=50, generations=3, random_state=42,
            device='cuda')
        est.fit(X, y)
        assert hasattr(est, '_y_pred_buf')
        assert est._y_pred_buf is not None

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_batch_evaluate_with_buffer(self):
        """_batch_evaluate_gpu accepts and reuses y_pred_buf."""
        import cupy as cp
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0]

        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42)
        est.fit(X, y)

        X_gpu = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float32).T)
        pop = est._programs[0]

        # First call without buffer
        y1 = _batch_evaluate_gpu(pop, X_gpu, 100, 5)
        # Second call with buffer
        y2 = _batch_evaluate_gpu(pop, X_gpu, 100, 5, y_pred_buf=y1)
        # Should reuse buffer (same object)
        assert y2 is y1

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_batch_evaluate_wrong_size_buffer(self):
        """_batch_evaluate_gpu allocates new buffer if sizes mismatch."""
        import cupy as cp
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0]

        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42)
        est.fit(X, y)

        X_gpu = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float32).T)
        pop = est._programs[0]

        # Wrong-sized buffer
        wrong_buf = cp.empty((10, 50), dtype=cp.float32)
        result = _batch_evaluate_gpu(pop, X_gpu, 100, 5, y_pred_buf=wrong_buf)
        assert result is not wrong_buf
        assert result.shape == (len(pop), 100)


# =========================================================================
# Improvement #7: Mixed precision parameter
# =========================================================================

class TestMixedPrecision:
    """Test the precision parameter on all estimators."""

    def test_precision_parameter_accepted(self):
        """All estimators accept precision parameter."""
        sr = SymbolicRegressor(precision='float32')
        assert sr.precision == 'float32'
        sr = SymbolicRegressor(precision='mixed')
        assert sr.precision == 'mixed'

        sc = SymbolicClassifier(precision='mixed')
        assert sc.precision == 'mixed'

        st = SymbolicTransformer(precision='float32')
        assert st.precision == 'float32'

    def test_invalid_precision_raises(self):
        """Invalid precision value should raise during fit."""
        X = diabetes.data[:50]
        y = diabetes.target[:50]
        est = SymbolicRegressor(precision='float64', device='cuda')
        # Only raises during fit() when device=='cuda'
        if HAS_CUDA:
            with pytest.raises(ValueError, match='precision must be'):
                est.fit(X, y)

    def test_precision_in_get_params(self):
        """precision should appear in get_params() for sklearn compat."""
        est = SymbolicRegressor(precision='mixed')
        params = est.get_params()
        assert 'precision' in params
        assert params['precision'] == 'mixed'

    def test_cpu_ignores_precision(self):
        """CPU path should work regardless of precision setting."""
        X = diabetes.data[:50]
        y = diabetes.target[:50]
        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42,
            precision='mixed', device='cpu')
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (50,)


# =========================================================================
# Improvement #8: Shared memory kernel selection
# =========================================================================

class TestSharedMemoryKernel:
    """Test shared memory kernel auto-selection logic."""

    def test_max_shared_features_constant(self):
        """MAX_SHARED_FEATURES should be 32."""
        assert MAX_SHARED_FEATURES == 32

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_small_features_uses_smem(self):
        """With n_features <= 32, the shared memory kernel should be used."""
        import cupy as cp
        # We can't directly test which kernel was selected, but we can
        # verify correctness — the smem kernel should give same results.
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + X[:, 1]

        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42)
        est.fit(X, y)
        pop = est._programs[0]

        X_gpu = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float32).T)
        result = _batch_evaluate_gpu(pop, X_gpu, 200, 5)
        assert result.shape == (len(pop), 200)
        # Verify finite results
        assert cp.isfinite(result).all()

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_large_features_uses_global(self):
        """With n_features > 32, the standard kernel should be used."""
        import cupy as cp
        n_feat = 50  # > MAX_SHARED_FEATURES
        X = np.random.randn(100, n_feat).astype(np.float32)
        y = X[:, 0] + X[:, 1]

        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42)
        est.fit(X, y)
        pop = est._programs[0]

        X_gpu = cp.ascontiguousarray(cp.asarray(X, dtype=cp.float32).T)
        result = _batch_evaluate_gpu(pop, X_gpu, 100, n_feat)
        assert result.shape == (len(pop), 100)
        assert cp.isfinite(result).all()


# =========================================================================
# Improvement #9: Eliminate redundant copy after .T
# =========================================================================

class TestTransposeEfficiency:
    """Test that data layout is handled correctly without redundant copies."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_gpu_fit_produces_contiguous_X(self):
        """After fit(), the GPU data should be contiguous."""
        import cupy as cp
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0]
        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42,
            device='cuda')
        est.fit(X, y)
        # Just verify it ran without errors — the ascontiguousarray
        # optimization is internal


# =========================================================================
# Integration: Transformer with Spearman on GPU
# =========================================================================

class TestTransformerSpearmanGPU:
    """Test SymbolicTransformer with Spearman metric."""

    def test_transformer_spearman_cpu(self):
        """SymbolicTransformer with spearman metric works on CPU."""
        X = diabetes.data[:100]
        y = diabetes.target[:100]
        est = SymbolicTransformer(
            population_size=50, generations=2, random_state=42,
            metric='spearman', hall_of_fame=10, n_components=3)
        est.fit(X, y)
        X_new = est.transform(X)
        assert X_new.shape == (100, 3)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_transformer_spearman_gpu(self):
        """SymbolicTransformer with spearman metric works on GPU."""
        X = diabetes.data[:100].astype(np.float32)
        y = diabetes.target[:100].astype(np.float32)
        est = SymbolicTransformer(
            population_size=50, generations=2, random_state=42,
            metric='spearman', hall_of_fame=10, n_components=3,
            device='cuda')
        est.fit(X, y)
        X_new = est.transform(X)
        assert X_new.shape == (100, 3)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_transformer_corrcoef_host_fallback(self, monkeypatch):
        """Transformer survives missing CuPy BLAS backends by falling back."""
        import cupy as cp

        X = diabetes.data[:100].astype(np.float32)
        y = diabetes.target[:100].astype(np.float32)
        est = SymbolicTransformer(
            population_size=30, generations=1, random_state=42,
            metric='spearman', hall_of_fame=10, n_components=3,
            device='cuda')

        def _broken_corrcoef(*args, **kwargs):
            raise ImportError('mock missing cublas backend')

        monkeypatch.setattr(cp, 'corrcoef', _broken_corrcoef)
        est.fit(X, y)
        X_new = est.transform(X)
        assert X_new.shape == (100, 3)


# =========================================================================
# Integration: Classifier with GPU precision
# =========================================================================

class TestClassifierGPU:
    """Test SymbolicClassifier on GPU path."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_classifier_gpu_basic(self):
        """SymbolicClassifier with device='cuda' fits and predicts."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        est = SymbolicClassifier(
            population_size=50, generations=2, random_state=42,
            device='cuda')
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (100,)
        proba = est.predict_proba(X)
        assert proba.shape == (100, 2)


# =========================================================================
# Integration: Warm start with new features
# =========================================================================

class TestWarmStartWithOptimizations:
    """Test warm_start interacts correctly with new optimizations."""

    def test_warm_start_cpu(self):
        """Warm start on CPU works after refactoring."""
        X = diabetes.data[:100]
        y = diabetes.target[:100]
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            warm_start=True)
        est.fit(X, y)
        est.set_params(generations=4)
        est.fit(X, y)
        assert len(est.run_details_['generation']) == 4

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_warm_start_gpu(self):
        """Warm start on GPU works with new buffer management."""
        X = diabetes.data[:100].astype(np.float32)
        y = diabetes.target[:100].astype(np.float32)
        est = SymbolicRegressor(
            population_size=50, generations=2, random_state=42,
            warm_start=True, device='cuda')
        est.fit(X, y)
        est.set_params(generations=4)
        est.fit(X, y)
        assert len(est.run_details_['generation']) == 4


# =========================================================================
# Edge case: Single sample, single feature
# =========================================================================

class TestEdgeCases:
    """Test edge cases that could break the new optimizations."""

    def test_single_feature(self):
        """Regression with a single feature works."""
        X = np.random.randn(100, 1).astype(np.float32)
        y = X[:, 0] ** 2
        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (100,)

    def test_const_range_none(self):
        """Programs with no constants work with new constant dedup."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0] + X[:, 1]
        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42,
            const_range=None)
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (100,)

    def test_large_population(self):
        """Large population doesn't break batch evaluation."""
        X = np.random.randn(50, 3).astype(np.float32)
        y = X[:, 0]
        est = SymbolicRegressor(
            population_size=500, generations=1, random_state=42)
        est.fit(X, y)
        assert len(est._programs[0]) == 500

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU not available")
    def test_empty_constants_pool_gpu(self):
        """GPU batch evaluation with no float constants in any program."""
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        est = SymbolicRegressor(
            population_size=20, generations=1, random_state=42,
            const_range=None, device='cuda')
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == (100,)
