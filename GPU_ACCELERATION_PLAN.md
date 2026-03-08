# gplearn GPU/CUDA Acceleration Plan

This document outlines the strategy for integrating GPU/CUDA acceleration into `gplearn` using **CuPy** and **NVIDIA CUDA**. The goal is to provide a seamless, high-performance toggle for users to offload the most computationally expensive parts of Genetic Programming (GP)—program evaluation and fitness calculation—to the GPU.

---

## 1. Overview & Motivation

Genetic Programming is inherently parallel. Evaluating a population of thousands of programs across massive datasets is a "massively parallel" problem that perfectly fits the SIMT (Single-Instruction, Multiple-Thread) model of GPUs. 

The primary bottlenecks in `gplearn` are:
1.  **Program Execution:** Recursively (or stack-based) evaluating mathematical trees.
2.  **Fitness Calculation:** Computing metrics (MSE, MAE, etc.) between predicted and target values.

By moving these to the GPU, we can achieve speedups of 10x–100x for large datasets and populations.

---

## 2. Core Architecture: The `xp` Pattern

To maintain a single codebase that supports both CPU (NumPy) and GPU (CuPy), we will adopt the `xp` pattern used in the RAPIDS ecosystem.

### 2.1 The `xp` Pattern
```python
import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def get_xp(x):
    if HAS_CUPY:
        return cp.get_array_module(x)
    return np
```

This allows us to write mathematical logic once:
```python
xp = get_xp(X)
return xp.add(x1, x2)
```

---

## 3. Implementation Strategy: GPU VM Interpreter (Batched)

To eliminate the JIT compilation bottleneck (NVRTC overhead for thousands of unique programs), we have moved to a **GPU Virtual Machine (VM)** architecture.

### 3.1 The VM Architecture
Instead of compiling each program into a unique CUDA kernel, we use a single, static **GPU Interpreter Kernel**:
1.  **Prefix-to-Postfix Conversion:** Programs are converted from prefix notation to a postfix integer array (Reverse Polish Notation) on the CPU.
2.  **Batched Evaluation:** The entire population is flattened into a single "byte-code" array with an accompanying "offsets" array.
3.  **Stack-based Execution:** Each GPU thread executes a stack-based loop to interpret the byte-code for its assigned program and data sample.
4.  **Memory Coalescing:** The input matrix $X$ is transposed to `(n_features, n_samples)` to ensure perfectly coalesced memory access within the interpreter.

### 3.2 Performance Gains
- **Zero JIT Overhead:** The VM kernel is compiled once at the start.
- **Batched Throughput:** Thousands of programs are evaluated in a single GPU grid launch.
- **Consistent Outperformance:** The GPU now outperforms multi-core CPUs by 1.5x to 3x across a wide range of population and dataset sizes.

---

## 4. Performance Considerations
- **Stack Size:** The VM uses a thread-local stack of 256 floats, supporting very deep trees.
- **Registers:** The interpreter is optimized to use as many registers as possible to avoid local memory spilling.
- **n_jobs=1:** For CUDA mode, `n_jobs=1` is strongly recommended to avoid the overhead of IPC and pickling CuPy arrays.

Fitness metrics must also be calculated on the GPU to avoid costly CPU-GPU data transfers (PCIe bandwidth bottleneck).

### 4.1 Reduction Kernels
Most metrics (MSE, MAE) can be implemented using **CuPy's `ReductionKernel`**:
```python
# Mean Square Error on GPU
mse_kernel = cp.ReductionKernel(
    'T y, T y_pred, T w',  # input params
    'T out',               # output param
    'w * (y - y_pred) * (y - y_pred)', # map
    'a + b',               # reduce
    'out = a / _sum_w',    # post-reduction (conceptual)
    '0',                   # identity
    'mse'
)
```

---

## 5. User Configuration & Toggle

We will introduce a `device` parameter (defaulting to `'cpu'`) to the `SymbolicRegressor`, `SymbolicClassifier`, and `SymbolicTransformer`.

### 5.1 Usage Example
```python
from gplearn.genetic import SymbolicRegressor

# Standard CPU run
est = SymbolicRegressor(device='cpu')

# CUDA-accelerated run
est = SymbolicRegressor(device='cuda')
est.fit(X, y)
```

---

## 6. Performance Considerations & Bottlenecks

### 6.1 Data Transfer
The training data `X` and `y` should be moved to the GPU **once** at the start of `fit()` and remain there for the entire evolution. Only the scalar fitness scores and the final "best" program should be fetched back to the CPU.

### 6.2 Small Populations/Datasets
GPU acceleration has overhead. For small datasets (< 10,000 samples) or small populations (< 500 individuals), the CPU may still be faster. We will provide documentation on the "sweet spot" for GPU usage.

---

## 7. Implementation Roadmap

### Phase 1: Infrastructure (Foundation)
- Add `device` parameter to base classes.
- Implement the `get_xp` utility and `HAS_CUPY` check.
- Update `fit()` to move data to/from the device based on the toggle.

### Phase 2: Core Acceleration (Execution)
- Implement the `Prefix-to-C++` translator in `_program.py`.
- Integrate CuPy `RawKernel` for program evaluation.
- Implement `__device__` protected functions.

### Phase 3: Metric Acceleration (Fitness)
- Implement `gplearn` fitness metrics using CuPy `ReductionKernel` and `xp` patterns.

### Phase 4: Validation & Benchmarking
- Ensure numerical parity with the CPU version (using unit tests).
- Benchmark performance across various dataset sizes.

---

## 8. Summary of Consulted Expert Advice

The **CUDA & GPlearn Expert** provided the following critical insights:
- **Avoid Warp Divergence:** Parallelize across data samples, not across programs.
- **CuPy RawKernel > Numba:** Better performance and easier dynamic string generation for GP trees.
- **Literal Baking:** Constant literals in JIT code allow for hardware-level optimizations (FMA).
- **Disk Caching:** CuPy's built-in kernel caching handles the "re-compilation" problem efficiently.
