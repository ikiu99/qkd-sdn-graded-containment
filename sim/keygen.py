"""Vectorised key generation.

Phase 2 reference: section 4.3 of phase2-build-spec.
Phase 1 section 1.4:  B_e(t+dt) = min(B_max, B_e(t) + R_e*dt - C_e(t))
Consumption C_e is applied by demand.SessionManager, not here.
"""
from __future__ import annotations

import numpy as np


def step(buffers: np.ndarray, R: np.ndarray, dt: float, B_max: float) -> None:
    """In place: buffers = clip(buffers + R*dt, 0, B_max).

    O(|E|) per step, no session traversal.  Key generated beyond B_max is lost
    (the buffer saturates); the runner measures the effective generation as the
    change in total buffer, so key conservation stays exact.
    """
    np.clip(buffers + R * dt, 0.0, B_max, out=buffers)
