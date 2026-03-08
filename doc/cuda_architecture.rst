.. _cuda_architecture:

CUDA Architecture
=================

``gplearn-CUDA`` utilizes a hybrid acceleration model to achieve maximum 
throughput across different stages of the Genetic Programming lifecycle.

The Hybrid Model
----------------

1. **Batched VM Interpreter (Evolution)**:
   During the ``fit()`` process, thousands of candidate programs are evaluated
   simultaneously. To eliminate the overhead of thousands of JIT compilation 
   calls per generation, ``gplearn-CUDA`` uses a static **Virtual Machine (VM)**
   kernel. This kernel interprets a compact postfix byte-code representation 
   of the entire population in a single GPU grid launch.

2. **JIT RawKernel (Prediction)**:
   Once the best individual is found, ``predict()`` and ``transform()`` utilize
   CuPy's ``RawKernel`` to JIT-compile that specific program into optimized 
   NVIDIA machine code. This ensures the final model runs at peak hardware 
   performance.

Memory Coalescing
-----------------

A key optimization in the CUDA backend is the automatic management of memory 
layouts. Standard datasets are typically stored in "row-major" format 
(Samples, Features). However, GPU warps perform best when accessing memory in 
a "coalesced" manner.

``gplearn-CUDA`` automatically transposes input matrices to a "feature-major" 
layout (Features, Samples) on the device. This ensures that when 32 threads in 
a warp evaluate the same program node across 32 different samples, they 
access contiguous memory addresses, maximizing bandwidth utilization.

VM Specifications
-----------------

- **Stack Depth**: The interpreter supports a thread-local stack of up to 256 
  floats, allowing for extremely deep and complex symbolic expressions.
- **Opcode Engine**: Supports all built-in mathematical primitives including 
  protected functions and trigonometric operations.
- **Fast Math**: The VM utilizes NVIDIA's ``-use_fast_math`` compiler 
  optimizations for accelerated transcendental functions.

Numerical Parity
----------------

The CUDA VM is rigorously tested to ensure bit-perfect (or near-perfect within 
floating point tolerance) numerical parity with the original scikit-learn 
compatible CPU implementation. All "protected" math logic (e.g., division by 
zero returning 1.0) is mirrored exactly in the CUDA C++ implementation.
