"""Phase 0 scaffold suite: the package imports and exposes its version.

Real behavior tests land in Phase 1 (test_adaptive_parity.py). These tests
exist so CI has a green suite to run against the scaffold.
"""

import ebbifold


def test_package_imports() -> None:
    assert ebbifold.__version__ == "0.0.1"
