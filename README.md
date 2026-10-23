# tl2c

TileLang → C/CUDA source translator. It emits the **generated C/CUDA source** for a
TileLang operator — not a binary — so downstream C++/Zig/Rust projects can compile
that source directly with their own toolchain.

## How it works

`tl2c.py` is a thin CLI over TileLang's existing "lower without device compile" path:

1. Loads your operator file (must define a `@tilelang.jit` kernel or a `PrimFunc`).
2. Binds the compile-time arguments via `JITImpl.get_tir(**args)`.
3. Runs `tilelang.lower(..., enable_device_compile=False)`.
4. Emits `artifact.kernel_source` — the hardware-native C/CUDA source.

No GPU is required and no device binary is produced: only source code. The CUDA arch
is pinned (`sm_80`) so codegen works headless.

## Install

Only the `tilelang` package is needed in Python (it pulls in its own deps):

```sh
pip install tilelang
```

## Usage

```
python tl2c.py <op.py> [-o out.cu] [--target cuda] [--args KEY=VALUE ...]
```

| Option | Description |
|---|---|
| `input` | TileLang operator `.py` file (must define a `@tilelang.jit` kernel or a `PrimFunc`) |
| `-o, --output` | output file (default: print to stdout) |
| `--target` | codegen target (default `cuda`; arch pinned to `sm_80`) |
| `--args` | compile-time arguments as `KEY=VALUE` pairs (see below) |

`--args` values are parsed as `int`/`float` where possible, otherwise passed as
strings (used for dtypes). A lazy-style kernel needs **all** of its compile-time
arguments (block sizes, `threads`, dtypes, `M`/`N`, ...); a missing one raises
TileLang's own `missing a required argument` / `Cannot find value for constexpr
variable` error.

## Example

`op.py` (a minimal, runnable operator):

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def elementwise_add(A, B, block_M, block_N, in_dtype, out_dtype, threads):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), in_dtype)
    B: T.Tensor((M, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), in_dtype)
        B_shared = T.alloc_shared((block_M, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.copy(A[by * block_M, bx * block_N], A_shared)
        T.copy(B[by * block_M, bx * block_N], B_shared)
        for y, x in T.Parallel(block_M, block_N):
            C_local[y, x] = A_shared[y, x] + B_shared[y, x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C
```

Emit the CUDA source to a file:

```sh
python tl2c.py op.py -o elementwise_add.cu \
  --args M=1024 N=1024 block_M=32 block_N=32 threads=128 \
         in_dtype=float32 out_dtype=float32
```

Or print to stdout:

```sh
python tl2c.py op.py \
  --args M=1024 N=1024 block_M=32 block_N=32 threads=128 \
         in_dtype=float32 out_dtype=float32
```

## Standalone executable (no Python required to run)

`build.sh` uses PyInstaller to bundle `tl2c.py` together with the TileLang
runtime (its own TVM, native libs, and codegen headers) into a single
`dist/tl2c/` directory. The resulting binary runs on a machine that has **no
Python and no tilelang** installed.

```sh
./build.sh                       # -> dist/tl2c/tl2c  (self-test included)
./dist/tl2c/tl2c op.py -o out.cu --args M=1024 N=1024 block_M=32 block_N=32 \
    threads=128 in_dtype=float32 out_dtype=float32
```

Notes on the bundle:

- **onedir, not onefile**: TileLang/TVm load native `.so` files at fixed
  package-relative paths during `import tilelang`; a self-extracting onefile
  would race that. The output is a `dist/tl2c/` folder, move it as a unit.
- **torch is not bundled**: `torch_stub.py` (installed by the runtime hook
  `_tl2c_hook.py` before the entry script) satisfies TileLang's unconditional
  `import torch`, so the ~700MB torch wheel stays out of the bundle. The
  source-lowering path this tool uses never touches a real tensor.
- Output is byte-identical to running `python tl2c.py` in a real env (the
  `build.sh` self-test emits an operator and checks it).

## Notes

- Only source is emitted; compile the output with your own `nvcc` / host C++ / Zig / Rust toolchain.
- The output is `extern "C" __global__ <name>_kernel(...)` plus any `#include`/`#define`
  prologue the codegen needs — it is the exact source TileLang would hand to the device
  compiler.
