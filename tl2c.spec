# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the tl2c standalone executable (onedir).

Build from the repo root with a python that has tilelang + PyInstaller:

    python -m PyInstaller tl2c.spec --clean --noconfirm

Design notes
------------
- onedir, not onefile: tvm/tvm_ffi/tilelang load native libs at fixed
  package-relative paths during ``import tilelang``; a self-extracting
  onefile would race that.
- torch (700MB+) is NOT bundled: the runtime hook ``_tl2c_hook.py`` installs
  ``torch_stub`` before the entry script runs, so tilelang's unconditional
  ``import torch`` resolves to the stub. ``torch`` is in ``excludes`` so the
  builder never follows it.
- tilelang bundles its own TVM under ``3rdparty/tvm/python`` and native libs
  under ``lib/``. ``collect_all`` collects the importable modules (compiled)
  plus every data/binary file in the package tree (headers, .so, the bundled
  TVM python files), preserving the layout under ``_MEIPASS``.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

repo_root = Path(SPECPATH).resolve()

import glob as _glob
import os as _os
import tilelang as _tl

_datas = []
_binaries = []
_hiddenimports = [
    # stdlib submodules PyInstaller does not always trace
    "logging.config",
    "configparser",
    "xml.dom",
    "xml.dom.minidom",
    "xml.dom.expatbuilder",
    "xml.dom.xmlbuilder",
]
for _pkg in ("tilelang", "tvm_ffi", "z3"):
    _d, _b, _h = collect_all(_pkg)
    _datas += _d
    _binaries += _b
    _hiddenimports += _h

# tilelang/lib/ holds top-level C-extension modules (importable as e.g.
# ``tilelang_cython_wrapper`` because tilelang/env.py puts this dir on
# sys.path). collect_all misses the ``.abi3.so`` ones, so collect every
# ``.so`` in tilelang/lib/ explicitly. Destination is the *directory*
# "tilelang/lib" (relative to the bundle root / site-packages).
_tl_root = _os.path.dirname(_tl.__file__)
_tl_lib = _os.path.join(_tl_root, "lib")
_have = {src for src, dst in _binaries}
for _so in sorted(_glob.glob(_os.path.join(_tl_lib, "*.so"))):
    if _so not in _have:
        _binaries.append((_so, "tilelang/lib"))
        _have.add(_so)

_datas += [(str(repo_root / "torch_stub.py"), ".")]  # installed by _tl2c_hook.py

a = Analysis(
    [str(repo_root / "tl2c.py")],
    pathex=[str(repo_root)],
    datas=_datas,
    binaries=_binaries,
    hiddenimports=_hiddenimports,
    hookspath=[str(repo_root)],
    hooksconfig={},
    runtime_hooks=[str(repo_root / "_tl2c_hook.py")],
    excludes=[
        "torch",
        "torch_c_dlpack_ext",
        "torchgen",
        "triton",
        "IPython",
        "pytest",
        "sphinx",
        "notebook",
        "jupyter",
        "matplotlib",
        "scipy",
        "pandas",
        "setuptools",
        "pkg_resources",
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir: binaries/data go to COLLECT, not the EXE
    name="tl2c",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="tl2c",
)
