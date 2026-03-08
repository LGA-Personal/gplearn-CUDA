import numpy as np
import time
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier, SymbolicTransformer

def run_benchmark(name, estimator_class, n_samples, n_features, pop_size, gens, **kwargs):
    print(f"\n>>> Scenario: {name} ({estimator_class.__name__})")
    print(f"    Config: {n_samples} samples, {n_features} features, {pop_size} pop, {gens} gens")
    
    # Generate synthetic data
    X = np.random.uniform(-1, 1, (n_samples, n_features)).astype(np.float32)
    if estimator_class == SymbolicClassifier:
        # Binary classification target
        y = (X[:, 0]**2 + X[:, 1]*X[:, 2] > 0.5).astype(int)
    else:
        # Regression target
        y = X[:, 0]**2 + X[:, 1]*X[:, 2]
        if n_features > 3:
            y -= np.sin(X[:, 3])
    y = y.astype(np.float32)

    # GPU Run
    print("    Running GPU (n_jobs=1)...")
    est_gpu = estimator_class(population_size=pop_size, generations=gens, 
                               random_state=42, device='cuda', verbose=1,
                               n_jobs=1, **kwargs)
    start = time.time()
    est_gpu.fit(X, y)
    
    # Final operation (predict or transform)
    if estimator_class == SymbolicTransformer:
        est_gpu.transform(X)
    else:
        est_gpu.predict(X)
        
    gpu_time = time.time() - start
    print(f"    GPU Total Time (fit + pred/trans): {gpu_time:.2f}s")

    # CPU Run
    print("    Running CPU (n_jobs=-1)...")
    est_cpu = estimator_class(population_size=pop_size, generations=gens, 
                               random_state=42, device='cpu', verbose=1,
                               n_jobs=-1, **kwargs)
    start = time.time()
    est_cpu.fit(X, y)
    
    if estimator_class == SymbolicTransformer:
        est_cpu.transform(X)
    else:
        est_cpu.predict(X)
        
    cpu_time = time.time() - start
    print(f"    CPU Total Time (fit + pred/trans): {cpu_time:.2f}s")
    
    speedup = cpu_time / gpu_time
    print(f"    >>> Speedup: {speedup:.2f}x")
    return speedup

def main():
    scenarios = [
        ("Regression Large", SymbolicRegressor, 250_000, 10, 1000, 10, {}),
        ("Classification Large", SymbolicClassifier, 250_000, 10, 1000, 10, {}),
        ("Transformer Large", SymbolicTransformer, 100_000, 10, 500, 10, {"hall_of_fame": 50, "n_components": 10}),
        ("High Dimensional Reg", SymbolicRegressor, 50_000, 50, 500, 10, {}),
        ("Complex Reg", SymbolicRegressor, 50_000, 5, 500, 10, {"function_set": ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt')})
    ]
    
    results = []
    for name, cls, n_s, n_f, pop, gens, kwargs in scenarios:
        try:
            speedup = run_benchmark(name, cls, n_s, n_f, pop, gens, **kwargs)
            results.append((f"{name} ({cls.__name__})", speedup))
        except Exception as e:
            print(f"    Scenario {name} failed: {e}")

    print("\n\n" + "="*40)
    print("FINAL MULTI-CLASS BENCHMARK SUMMARY")
    print("="*40)
    for name, speedup in results:
        print(f"{name:<35}: {speedup:>6.2f}x")

if __name__ == "__main__":
    main()
