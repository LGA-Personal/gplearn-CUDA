"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from collections import OrderedDict
from copy import copy
import struct

import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
from .utils import check_random_state, get_xp


# Bounded LRU cache for compiled CUDA kernels to avoid GPU memory exhaustion
_CUDA_KERNEL_CACHE_MAX_SIZE = 1024
_CUDA_KERNEL_CACHE = OrderedDict()


def _cache_kernel(key, kernel):
    """Add a kernel to the bounded cache, evicting oldest if full."""
    if key in _CUDA_KERNEL_CACHE:
        _CUDA_KERNEL_CACHE.move_to_end(key)
        return
    if len(_CUDA_KERNEL_CACHE) >= _CUDA_KERNEL_CACHE_MAX_SIZE:
        _CUDA_KERNEL_CACHE.popitem(last=False)
    _CUDA_KERNEL_CACHE[key] = kernel


def clear_kernel_cache():
    """Clear the CUDA kernel cache. Call to free GPU memory."""
    _CUDA_KERNEL_CACHE.clear()


def _float_to_key(val):
    """Convert a float to a bit-exact bytes key for deduplication.

    Handles NaN, -0.0, and subnormals correctly unlike float equality.
    """
    return struct.pack('f', val)

# Opcode mapping for GPU VM
_OPCODES = {
    'add': 0, 'sub': 1, 'mul': 2, 'div': 3,
    'sqrt': 4, 'log': 5, 'abs': 6, 'neg': 7,
    'inv': 8, 'max': 9, 'min': 10,
    'sin': 11, 'cos': 12, 'tan': 13, 'sig': 14
}

_VM_KERNEL_SOURCE = """
extern "C" {
    __device__ inline float protected_div(float num, float den) {
        if (fabsf(den) < 0.001f) return 1.0f;
        return __fdividef(num, den);
    }
    __device__ inline float protected_log(float val) {
        float abs_val = fabsf(val);
        if (abs_val < 0.001f) return 0.0f;
        return __logf(abs_val);
    }
    __device__ inline float protected_sqrt(float val) {
        return __fsqrt_rn(fabsf(val));
    }
    __device__ inline float protected_inv(float val) {
        if (fabsf(val) < 0.001f) return 0.0f;
        return __frcp_rn(val);
    }
    __device__ inline float sigmoid(float x) {
        return __fdividef(1.0f, (1.0f + __expf(-x)));
    }

    // Standard kernel: reads float32 from global memory
    __global__ void evaluate_population_vm(
        const int* __restrict__ opcodes,
        const float* __restrict__ constants,
        const float* __restrict__ X,          // (n_features, n_samples)
        float* __restrict__ y_pred,           // (n_programs, n_samples)
        const int* __restrict__ prog_offsets,
        int n_samples,
        int n_features)
    {
        int prog_idx = blockIdx.x;
        int sample_idx = threadIdx.x + blockIdx.y * blockDim.x;

        if (sample_idx >= n_samples) return;

        int start_op = prog_offsets[prog_idx];
        int end_op   = prog_offsets[prog_idx + 1];

        float stack[256];
        int sp = 0;

        for (int i = start_op; i < end_op; ++i) {
            int op = opcodes[i];

            if (op >= 20000) {
                // Variable
                int feat_idx = op - 20000;
                if (feat_idx < n_features && sp < 256) {
                    stack[sp++] = X[feat_idx * n_samples + sample_idx];
                }
            } else if (op >= 10000) {
                // Constant
                if (sp < 256) {
                    stack[sp++] = constants[op - 10000];
                }
            } else {
                // Function
                if (sp < 1) continue; // Safety
                float a, b;
                switch(op) {
                    case 0: // ADD
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a + b; break;
                    case 1: // SUB
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a - b; break;
                    case 2: // MUL
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a * b; break;
                    case 3: // DIV
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = protected_div(a, b); break;
                    case 4: // SQRT
                        a = stack[--sp]; stack[sp++] = protected_sqrt(a); break;
                    case 5: // LOG
                        a = stack[--sp]; stack[sp++] = protected_log(a); break;
                    case 6: // ABS
                        a = stack[--sp]; stack[sp++] = fabsf(a); break;
                    case 7: // NEG
                        a = stack[--sp]; stack[sp++] = -a; break;
                    case 8: // INV
                        a = stack[--sp]; stack[sp++] = protected_inv(a); break;
                    case 9: // MAX
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fmaxf(a, b); break;
                    case 10: // MIN
                        if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fminf(a, b); break;
                    case 11: // SIN
                        a = stack[--sp]; stack[sp++] = __sinf(a); break;
                    case 12: // COS
                        a = stack[--sp]; stack[sp++] = __cosf(a); break;
                    case 13: // TAN
                        a = stack[--sp]; stack[sp++] = __tanf(a); break;
                    case 14: // SIG
                        a = stack[--sp]; stack[sp++] = sigmoid(a); break;
                }
            }
        }
        if (sp > 0) {
            y_pred[prog_idx * n_samples + sample_idx] = stack[sp - 1];
        } else {
            y_pred[prog_idx * n_samples + sample_idx] = 0.0f;
        }
    }

    // Shared-memory kernel: caches feature data in shared memory for
    // problems with small n_features (<=MAX_SHARED_FEATURES).
    // Each block cooperatively loads feature columns into shared memory,
    // then the VM reads from shared memory instead of global memory.
    // This provides ~10x bandwidth improvement for repeated feature access.
    __global__ void evaluate_population_vm_smem(
        const int* __restrict__ opcodes,
        const float* __restrict__ constants,
        const float* __restrict__ X,          // (n_features, n_samples)
        float* __restrict__ y_pred,           // (n_programs, n_samples)
        const int* __restrict__ prog_offsets,
        int n_samples,
        int n_features)
    {
        int prog_idx = blockIdx.x;
        int sample_idx = threadIdx.x + blockIdx.y * blockDim.x;

        // Shared memory: one float per feature per thread in the block
        // Dynamically sized via launch parameter
        extern __shared__ float smem[];

        // Load feature data into shared memory
        if (sample_idx < n_samples) {
            for (int f = 0; f < n_features; f++) {
                smem[f * blockDim.x + threadIdx.x] = X[f * n_samples + sample_idx];
            }
        }
        __syncthreads();

        if (sample_idx >= n_samples) return;

        int start_op = prog_offsets[prog_idx];
        int end_op   = prog_offsets[prog_idx + 1];

        float stack[256];
        int sp = 0;

        for (int i = start_op; i < end_op; ++i) {
            int op = opcodes[i];

            if (op >= 20000) {
                int feat_idx = op - 20000;
                if (feat_idx < n_features && sp < 256) {
                    // Read from shared memory instead of global
                    stack[sp++] = smem[feat_idx * blockDim.x + threadIdx.x];
                }
            } else if (op >= 10000) {
                if (sp < 256) {
                    stack[sp++] = constants[op - 10000];
                }
            } else {
                if (sp < 1) continue;
                float a, b;
                switch(op) {
                    case 0: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a + b; break;
                    case 1: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a - b; break;
                    case 2: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a * b; break;
                    case 3: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = protected_div(a, b); break;
                    case 4: a = stack[--sp]; stack[sp++] = protected_sqrt(a); break;
                    case 5: a = stack[--sp]; stack[sp++] = protected_log(a); break;
                    case 6: a = stack[--sp]; stack[sp++] = fabsf(a); break;
                    case 7: a = stack[--sp]; stack[sp++] = -a; break;
                    case 8: a = stack[--sp]; stack[sp++] = protected_inv(a); break;
                    case 9: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fmaxf(a, b); break;
                    case 10: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fminf(a, b); break;
                    case 11: a = stack[--sp]; stack[sp++] = __sinf(a); break;
                    case 12: a = stack[--sp]; stack[sp++] = __cosf(a); break;
                    case 13: a = stack[--sp]; stack[sp++] = __tanf(a); break;
                    case 14: a = stack[--sp]; stack[sp++] = sigmoid(a); break;
                }
            }
        }
        if (sp > 0) {
            y_pred[prog_idx * n_samples + sample_idx] = stack[sp - 1];
        } else {
            y_pred[prog_idx * n_samples + sample_idx] = 0.0f;
        }
    }

    // Mixed-precision kernel: reads float16 input, accumulates in float32.
    // Provides up to 2x memory bandwidth improvement for large datasets.
    __global__ void evaluate_population_vm_fp16(
        const int* __restrict__ opcodes,
        const short* __restrict__ constants_fp16,
        const short* __restrict__ X_fp16,     // (n_features, n_samples) as float16
        float* __restrict__ y_pred,           // (n_programs, n_samples)
        const int* __restrict__ prog_offsets,
        int n_samples,
        int n_features)
    {
        int prog_idx = blockIdx.x;
        int sample_idx = threadIdx.x + blockIdx.y * blockDim.x;

        if (sample_idx >= n_samples) return;

        int start_op = prog_offsets[prog_idx];
        int end_op   = prog_offsets[prog_idx + 1];

        float stack[256];
        int sp = 0;

        for (int i = start_op; i < end_op; ++i) {
            int op = opcodes[i];

            if (op >= 20000) {
                int feat_idx = op - 20000;
                if (feat_idx < n_features && sp < 256) {
                    // Read float16 and convert to float32 for accumulation
                    stack[sp++] = __half2float(*reinterpret_cast<const __half*>(
                        &X_fp16[feat_idx * n_samples + sample_idx]));
                }
            } else if (op >= 10000) {
                if (sp < 256) {
                    stack[sp++] = __half2float(*reinterpret_cast<const __half*>(
                        &constants_fp16[op - 10000]));
                }
            } else {
                if (sp < 1) continue;
                float a, b;
                switch(op) {
                    case 0: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a + b; break;
                    case 1: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a - b; break;
                    case 2: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = a * b; break;
                    case 3: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = protected_div(a, b); break;
                    case 4: a = stack[--sp]; stack[sp++] = protected_sqrt(a); break;
                    case 5: a = stack[--sp]; stack[sp++] = protected_log(a); break;
                    case 6: a = stack[--sp]; stack[sp++] = fabsf(a); break;
                    case 7: a = stack[--sp]; stack[sp++] = -a; break;
                    case 8: a = stack[--sp]; stack[sp++] = protected_inv(a); break;
                    case 9: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fmaxf(a, b); break;
                    case 10: if (sp < 2) break; b = stack[--sp]; a = stack[--sp]; stack[sp++] = fminf(a, b); break;
                    case 11: a = stack[--sp]; stack[sp++] = __sinf(a); break;
                    case 12: a = stack[--sp]; stack[sp++] = __cosf(a); break;
                    case 13: a = stack[--sp]; stack[sp++] = __tanf(a); break;
                    case 14: a = stack[--sp]; stack[sp++] = sigmoid(a); break;
                }
            }
        }
        if (sp > 0) {
            y_pred[prog_idx * n_samples + sample_idx] = stack[sp - 1];
        } else {
            y_pred[prog_idx * n_samples + sample_idx] = 0.0f;
        }
    }
}
"""

# Maximum number of features for which the shared memory kernel is used.
# Above this threshold, shared memory requirements exceed typical GPU limits.
# 32 features * 256 threads * 4 bytes = 32KB, well within the 48KB default.
MAX_SHARED_FEATURES = 32

_VM_MODULE = None


def _batch_evaluate_gpu(programs, X, n_samples, n_features, y_pred_buf=None,
                        precision='float32'):
    """Evaluate a population of programs in one batch using the GPU VM.

    Parameters
    ----------
    programs : list of _Program
        The programs to evaluate.
    X : cupy.ndarray
        Input data, shape (n_features, n_samples).
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    y_pred_buf : cupy.ndarray, optional
        Pre-allocated output buffer of shape (n_programs, n_samples).
        If provided and correctly sized, it is reused to avoid allocation.
    precision : str, optional (default='float32')
        Precision mode: 'float32' or 'mixed' (float16 reads, float32 accumulation).

    Returns
    -------
    y_pred : cupy.ndarray, shape (n_programs, n_samples)
    """
    import cupy as cp
    global _VM_MODULE
    if _VM_MODULE is None:
        _VM_MODULE = cp.RawModule(code=_VM_KERNEL_SOURCE,
                                  options=('-use_fast_math',))

    all_opcodes = []
    offsets = [0]
    constants_pool = []
    constants_map = {}

    for prog in programs:
        postfix = prog.to_postfix()
        prog_opcodes = []
        for node in postfix:
            if not isinstance(node, (int, np.integer)):
                # Constant — use bit-exact key for deduplication
                val = float(node)
                key = _float_to_key(val)
                if key not in constants_map:
                    constants_map[key] = len(constants_pool)
                    constants_pool.append(val)
                prog_opcodes.append(10000 + constants_map[key])
            elif node >= 1000:
                # Variable
                prog_opcodes.append(20000 + (node - 1000))
            else:
                # Opcode
                prog_opcodes.append(node)

        all_opcodes.extend(prog_opcodes)
        offsets.append(len(all_opcodes))

    # Move to GPU
    d_opcodes = cp.array(all_opcodes, dtype=cp.int32)
    d_offsets = cp.array(offsets, dtype=cp.int32)

    # Handle empty constants pool
    if not constants_pool:
        d_constants = cp.zeros(1, dtype=cp.float32)
    else:
        d_constants = cp.array(constants_pool, dtype=cp.float32)

    n_programs = len(programs)

    # Reuse pre-allocated buffer if provided and correctly sized
    if (y_pred_buf is not None and y_pred_buf.shape == (n_programs, n_samples)
            and y_pred_buf.dtype == cp.float32):
        y_pred = y_pred_buf
    else:
        y_pred = cp.empty((n_programs, n_samples), dtype=cp.float32)

    threads_per_block = 256
    blocks_per_sample = (int(n_samples) + threads_per_block - 1) // threads_per_block

    # Choose kernel variant based on precision and feature count
    if precision == 'mixed':
        kernel = _VM_MODULE.get_function('evaluate_population_vm_fp16')
        # Convert data to float16 for bandwidth savings
        X_fp16 = X.astype(cp.float16)
        d_constants_fp16 = d_constants.astype(cp.float16)
        kernel((n_programs, blocks_per_sample), (threads_per_block,),
               (d_opcodes, d_constants_fp16, X_fp16, y_pred, d_offsets,
                int(n_samples), int(n_features)))
    elif n_features <= MAX_SHARED_FEATURES:
        # Use shared memory kernel for small feature counts
        kernel = _VM_MODULE.get_function('evaluate_population_vm_smem')
        smem_bytes = n_features * threads_per_block * 4  # 4 bytes per float
        kernel((n_programs, blocks_per_sample), (threads_per_block,),
               (d_opcodes, d_constants, X, y_pred, d_offsets,
                int(n_samples), int(n_features)),
               shared_mem=smem_bytes)
    else:
        # Standard global memory kernel
        kernel = _VM_MODULE.get_function('evaluate_population_vm')
        kernel((n_programs, blocks_per_sample), (threads_per_block,),
               (d_opcodes, d_constants, X, y_pred, d_offsets,
                int(n_samples), int(n_features)))

    return y_pred


def _batch_execute_gpu(programs, X, precision='float32'):
    """Execute a list of programs in one batch on the GPU.

    Parameters
    ----------
    programs : list of _Program
        The programs to execute.

    X : cupy.ndarray, shape = (n_features, n_samples)
        The input data on the GPU.

    precision : str, optional (default='float32')
        Precision mode: 'float32' or 'mixed'.

    Returns
    -------
    y_pred : cupy.ndarray, shape = (n_programs, n_samples)
        The results of executing each program.
    """
    n_features, n_samples = X.shape
    return _batch_evaluate_gpu(programs, X, n_samples, n_features,
                               precision=precision)


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    device : str, optional (default='cpu')
        The device on which to perform the evolution.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 device='cpu',
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.device = device
        self.random_state = random_state
        self._cuda_kernel = None
        self._postfix_cache = None
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def to_postfix(self):
        """Convert the prefix program to a postfix integer array (Reverse Polish Notation).

        Uses a cached result when available. The cache is valid for the
        lifetime of a program instance since the program list is never
        mutated in-place — genetic operations always create new lists.

        Returns
        -------
        postfix : list
            The postfix representation where:
            - int < 1000: Opcodes for functions
            - int >= 1000: Indices for features (index = op - 1000)
            - float/np.floating: Constant values
        """
        if self._postfix_cache is not None:
            return self._postfix_cache

        tmp_stack = []
        for node in reversed(self.program):
            if not isinstance(node, _Function):
                # Terminal
                if isinstance(node, (int, np.integer)):
                    tmp_stack.append([1000 + node])
                else:
                    # Constant (float or numpy floating)
                    tmp_stack.append([node])
            else:
                # Operator: Pop args and reverse them to maintain correct order [arg1, arg2, op]
                args = [tmp_stack.pop() for _ in range(node.arity)][::-1]
                # Combine args then the operator
                new_expr = []
                for arg in args:
                    new_expr.extend(arg)
                new_expr.append(_OPCODES[node.name])
                tmp_stack.append(new_expr)

        self._postfix_cache = tmp_stack[0]
        return self._postfix_cache

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    _CUDA_HEADER = """
    extern "C" {
        __device__ inline float protected_div(float x, float y) {
            return (fabsf(y) > 0.001f) ? (x / y) : 1.0f;
        }
        __device__ inline float protected_sqrt(float x) {
            return sqrtf(fabsf(x));
        }
        __device__ inline float protected_log(float x) {
            return (fabsf(x) > 0.001f) ? logf(fabsf(x)) : 0.0f;
        }
        __device__ inline float protected_inv(float x) {
            return (fabsf(x) > 0.001f) ? (1.0f / x) : 0.0f;
        }
        __device__ inline float sigmoid(float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __global__ void evaluate_program(const float* X, float* y_pred, int n_samples, int n_features) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < n_samples) {
                y_pred[tid] = %s;
            }
        }
    }
    """

    def _compile_cuda_kernel(self):
        """Translate the prefix program to C++ and JIT-compile a CUDA kernel."""
        import cupy as cp

        # Use the string representation as the cache key (bounded LRU cache)
        cache_key = str(self)
        if cache_key in _CUDA_KERNEL_CACHE:
            _CUDA_KERNEL_CACHE.move_to_end(cache_key)
            self._cuda_kernel = _CUDA_KERNEL_CACHE[cache_key]
            return

        # Mapping of gplearn functions to C++ device functions
        cpp_map = {
            'add': '(%s + %s)',
            'sub': '(%s - %s)',
            'mul': '(%s * %s)',
            'div': 'protected_div(%s, %s)',
            'sqrt': 'protected_sqrt(%s)',
            'log': 'protected_log(%s)',
            'abs': 'fabsf(%s)',
            'neg': '-(%s)',
            'inv': 'protected_inv(%s)',
            'max': 'fmaxf(%s, %s)',
            'min': 'fminf(%s, %s)',
            'sin': 'sinf(%s)',
            'cos': 'cosf(%s)',
            'tan': 'tanf(%s)',
            'sig': 'sigmoid(%s)'
        }

        # Translate prefix notation to C++ expression string
        expr_stack = []
        for node in reversed(self.program):
            if isinstance(node, _Function):
                args = [expr_stack.pop() for _ in range(node.arity)]
                expr_stack.append(cpp_map[node.name] % tuple(args))
            elif isinstance(node, int):
                # Variable access: Coalesced indexing X[feat * n_samples + tid]
                expr_stack.append('X[%d * n_samples + tid]' % node)
            else:
                # Constant literal
                expr_stack.append('%ff' % node)

        cpp_expr = expr_stack[0]
        full_source = self._CUDA_HEADER % cpp_expr

        # JIT-compile using CuPy
        module = cp.RawModule(code=full_source, options=('--std=c++11',))
        self._cuda_kernel = module.get_function('evaluate_program')
        _cache_kernel(cache_key, self._cuda_kernel)

    def execute(self, X, stream=None):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        stream : cupy.cuda.Stream, optional (default=None)
            CUDA stream to use for execution.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        if self.device == 'cuda':
            import cupy as cp
            # Ensure X is on GPU and in correct shape (n_features, n_samples)
            if not isinstance(X, cp.ndarray):
                X = cp.asarray(X, dtype=cp.float32)

            # Determine layout: predict/transform always receives standard
            # (n_samples, n_features) input, so we must transpose.
            # During fit(), X is already (n_features, n_samples).
            # We check shape[1] == n_features as the reliable indicator
            # that X is in standard layout and needs transposing.
            # This avoids the broken heuristic that fails when
            # n_samples == n_features (square matrices).
            if X.ndim == 2 and X.shape[1] == self.n_features and X.shape[0] != self.n_features:
                X = cp.ascontiguousarray(X.T)
            elif X.ndim == 2 and X.shape[0] != self.n_features:
                # Ambiguous case (e.g. square matrix): assume standard layout
                X = cp.ascontiguousarray(X.T)
            
            n_features, n_samples = X.shape
            y_hats = cp.empty(n_samples, dtype=cp.float32)

            if self._cuda_kernel is None:
                self._compile_cuda_kernel()

            # Grid-stride loop configuration
            threads_per_block = 256
            blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
            
            if stream is not None:
                with stream:
                    self._cuda_kernel((blocks_per_grid,), (threads_per_block,),
                                      (X, y_hats, n_samples, n_features))
            else:
                self._cuda_kernel((blocks_per_grid,), (threads_per_block,),
                                  (X, y_hats, n_samples, n_features))
            return y_hats

        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight, stream=None):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        stream : cupy.cuda.Stream, optional (default=None)
            CUDA stream to use for execution.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X, stream=stream)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
