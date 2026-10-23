#!/usr/bin/env python3
"""tl2c: turn a TileLang Python operator into hardware-native C/CUDA source.

Thin CLI wrapper over TileLang's existing "lower() without device compile"
path.  It does NOT compile to a binary -- it emits the generated C/CUDA source
that downstream C++/Zig/Rust projects compile as source.

Usage:
    python tl2c.py <op.py> [-o out.cu] [--target cuda] [--args M=1024 ...]

The input operator file must define a @tilelang.jit kernel (or a PrimFunc).
Compile-time arguments (e.g. M, N, K, block_M) are passed via --args KEY=VALUE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_module(path: str):
    import importlib.util
    from types import ModuleType

    spec = importlib.util.spec_from_file_location("tl2c_input", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load input file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _discover_prim_func(module):
    """Find the first @tilelang.jit / PrimFunc in the module."""
    from tilelang.jit import JITImpl
    from tilelang.language.eager import PrimFunc

    for obj in vars(module).values():
        if isinstance(obj, JITImpl):
            return obj
        if isinstance(obj, PrimFunc):
            return obj
    raise RuntimeError("no @tilelang.jit kernel or PrimFunc found in input")


def _parse_args(args_list: list[str]) -> dict:
    """Parse KEY=VALUE compile-time args (int/float where possible)."""
    out = {}
    for item in args_list:
        if "=" not in item:
            raise ValueError(f"--args entries must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tl2c",
        description="Generate hardware-native C/CUDA source from a TileLang operator.",
    )
    parser.add_argument("input", help="TileLang operator .py file")
    parser.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    parser.add_argument("--target", default="cuda", help="codegen target (default cuda)")
    parser.add_argument(
        "--args", nargs="*", default=[],
        help="compile-time args, e.g. M=1024 N=1024 block_M=128",
    )
    args = parser.parse_args(argv)

    module = _load_module(args.input)
    func = _discover_prim_func(module)
    compile_args = _parse_args(args.args)

    # For JITImpl, pass compile-time args to get_tir(); for a bare PrimFunc, call directly.
    from tilelang.jit import JITImpl
    from tilelang.language.eager import PrimFunc
    if isinstance(func, JITImpl):
        prim_func = func.get_tir(**compile_args)
    elif isinstance(prim_func := func, PrimFunc):
        prim_func = prim_func
    else:
        raise RuntimeError(f"unsupported kernel object: {type(func)}")

    # Lower without device compile -> generated C/CUDA source string.
    import tilelang
    from tvm.target import Target

    target = args.target
    if target == "cuda":
        # Pin arch so no GPU is required for codegen.
        target = {"kind": "cuda", "arch": "sm_80"}
    resolved = target if isinstance(target, Target) else Target(target)
    with tilelang.transform.PassContext(opt_level=3), resolved:
        artifact = tilelang.lower(prim_func, target=resolved, enable_device_compile=False)
    source = str(artifact.kernel_source)
    if not source.strip():
        print("tl2c: lower() produced empty kernel_source", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(source)
        print(f"tl2c: wrote {len(source)} bytes -> {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
