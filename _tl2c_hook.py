"""PyInstaller runtime hook for tl2c (runs before the entry script).

Installs the minimal torch stub so tilelang's unconditional ``import torch``
resolves without the 700MB torch wheel, and pins the bundled-data paths.
``torch_stub.py`` is shipped next to the executable by the spec.
"""

import os
import sys
from pathlib import Path

_base = Path(getattr(sys, "_MEIPASS", None) or Path(sys.argv[0]).resolve().parent)
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

import torch_stub

torch_stub.install()

# Pin bundled codegen headers / native libs (tilelang auto-detects these from
# its package dir; the env vars make it explicit and robust).
tl_src = _base / "tilelang" / "src"
if tl_src.is_dir():
    os.environ.setdefault("TL_TEMPLATE_PATH", str(tl_src))
tl_lib = _base / "tilelang" / "lib"
if tl_lib.is_dir():
    os.environ.setdefault("TVM_LIBRARY_PATH", str(tl_lib))
    # The tilelang/lib dir holds top-level extension modules
    # (e.g. tilelang_cython_wrapper.abi3.so). In a normal install,
    # tilelang/env.py inserts this dir onto sys.path; PyInstaller's frozen
    # importer does not scan package subdirs for .so, so add it here.
    if str(tl_lib) not in sys.path:
        sys.path.insert(0, str(tl_lib))
