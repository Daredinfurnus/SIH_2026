"""
Validation helpers for SVR feature vectors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def validate_feature_vector(
    vector: List[float],
    expected_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate a feature vector for SVR consumption.

    Returns a dict with:
        valid: bool
        issues: list[str] — human-readable problems found
        shape_ok: bool — correct length
        dtype_ok: bool — all numeric
        has_nan: bool
        has_inf: bool
        has_none: bool
    """
    issues: list[str] = []
    shape_ok = True
    dtype_ok = True
    has_nan = False
    has_inf = False
    has_none = False

    if expected_dim is not None and len(vector) != expected_dim:
        issues.append(f"Wrong dimension: got {len(vector)}, expected {expected_dim}")
        shape_ok = False

    if len(vector) == 0:
        issues.append("Empty feature vector")
        shape_ok = False
        return {
            "valid": False,
            "issues": issues,
            "shape_ok": False,
            "dtype_ok": False,
            "has_nan": False,
            "has_inf": False,
            "has_none": False,
        }

    for i, v in enumerate(vector):
        if v is None:
            has_none = True
            issues.append(f"Feature {i} is None")
            dtype_ok = False
            continue
        if not isinstance(v, (int, float)):
            issues.append(f"Feature {i} ({FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else '?'}): non-numeric type {type(v).__name__} [{v}]")
            dtype_ok = False
            continue
        if isinstance(v, float):
            if v != v:   # NaN
                has_nan = True
                issues.append(f"Feature {i} ({FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else '?'}): NaN")
            elif v == float('inf') or v == float('-inf'):
                has_inf = True
                issues.append(f"Feature {i} ({FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else '?'}): Inf")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "has_none": has_none,
    }
