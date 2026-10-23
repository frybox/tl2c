"""Minimal fake ``torch`` module injected before ``import tilelang``.

TileLang's ``__init__`` unconditionally does ``import torch`` (to preload it for
dlopen).  All the pure source-lowering path (``get_tir`` + ``lower`` without
device compile) needs from torch is a handful of dtype *values* used as dict
keys by ``tvm_ffi``'s dtype mapping.  We provide those; everything else raises
a clear error so nobody silently gets "works" behaviour from the stub.
"""

from __future__ import annotations

import os
import sys
import types

__all__ = ["install", "_TorchDtype"]


class _TorchDtype:
    """Stand-in for ``torch.dtype`` instances: hashable, comparable, printable."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(("tl2c-torch-stub.dtype", self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TorchDtype) and other.name == self.name

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"torch.{self.name}"


# Exactly the attributes tvm_ffi/cython/dtype.pxi reads at import time.
_DTYPE_NAMES = (
    "int8",
    "short",
    "int16",
    "int32",
    "int",
    "int64",
    "long",
    "uint8",
    "float16",
    "half",
    "float32",
    "float",
    "float64",
    "double",
    "bfloat16",
    "bool",
)


class _DeviceStub:
    """Class stand-in for ``torch.device`` (used in type annotations only)."""


class _TensorStub:
    """Class stand-in for ``torch.Tensor``; nothing real is an instance of it.

    Carries the DLPack-exchange dunder markers so ``tvm_ffi``/
    ``torch_c_dlpack_ext`` short-circuit (``hasattr`` early return) instead of
    probing the torch version and loading a version-specific native addon.
    """

    __c_dlpack_exchange_api__ = None
    __dlpack_c_exchange_api__ = None


class _StreamStub:
    """Class stand-in for ``torch.cuda.Stream`` (annotations only)."""


def _make_torch_module() -> types.ModuleType:
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "0.0.0"  # valid PEP440; torch_c_dlpack_ext parses it
    torch_mod.__spec__ = None
    torch_mod.Tensor = _TensorStub

    version = types.ModuleType("torch.version")
    # No ``hip`` attribute -> not detected as ROCm; ``cuda`` is None -> the
    # tvm_ffi CUDA-dtype auto-detect branch is skipped (it needs a real GPU).
    version.cuda = None
    version.hip = None
    torch_mod.version = version
    sys.modules["torch.version"] = version

    torch_mod.dtype = _TorchDtype
    torch_mod.device = _DeviceStub
    torch_mod.cuda = types.ModuleType("torch.cuda")
    torch_mod.cuda.is_available = lambda *a, **k: False
    torch_mod.cuda.Stream = _StreamStub
    sys.modules["torch.cuda"] = torch_mod.cuda

    for name in _DTYPE_NAMES:
        setattr(torch_mod, name, _TorchDtype(name))

    return torch_mod


def install() -> None:
    """Put the stub torch into sys.modules (idempotent).

    Also sets ``TVM_FFI_DISABLE_TORCH_C_DLPACK=1`` so tvm_ffi skips loading the
    torch_c_dlpack_ext native addon entirely (we never pass tensors).
    """
    os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    sys.modules["torch"] = _make_torch_module()
