"""Validation and projected geometry for a prescribed cruise ground track."""

from typing import Any

import casadi as ca
import openap.casadi as oc

import numpy as np


def validate_track(lat: Any, lon: Any, origin, destination):
    """Copy coordinates and snap nearby endpoints to the flight boundaries."""
    lat = np.array(lat, dtype=float, copy=True)
    lon = np.array(lon, dtype=float, copy=True)
    if lat.ndim != 1 or lon.ndim != 1 or lat.shape != lon.shape:
        raise ValueError("track latitude and longitude must be matching 1D arrays")
    if lat.size < 4:
        raise ValueError("track must contain at least four points for cubic splines")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("track coordinates must be finite")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("track coordinates must be valid latitude/longitude degrees")
    d0 = float(oc.geo.distance(lat[0], lon[0], *origin))
    dn = float(oc.geo.distance(lat[-1], lon[-1], *destination))
    if max(d0, dn) > 1000:
        raise ValueError(
            "track endpoints do not match origin/destination "
            f"(off by {d0 / 1000:.1f} km / {dn / 1000:.1f} km)"
        )
    lat[0], lon[0] = origin
    lat[-1], lon[-1] = destination
    return lat, lon


def project_track(lat, lon, proj):
    """Build arc-length splines in the projection used by the current solve."""
    x, y = proj(lon, lat)
    segments = np.hypot(np.diff(x), np.diff(y))
    if not np.all(np.isfinite(segments)) or np.any(segments <= 1e-6):
        raise ValueError("track must not contain duplicate consecutive points")
    s = np.concatenate([[0.0], np.cumsum(segments)])
    return (
        ca.interpolant("x_ref", "bspline", [s.tolist()], x.tolist()),
        ca.interpolant("y_ref", "bspline", [s.tolist()], y.tolist()),
        float(s[-1]),
    )


def constrain_track(transcription, track_ref, tolerance_m):
    """Pin boundary nodes and bound interior samples near an ordered spline."""
    opti = transcription.opti
    x_ref, y_ref, length = track_ref
    nodes = len(transcription.X) - 1
    progress = opti.variable(nodes + 1)
    opti.subject_to(opti.bounded(0, progress, 1))
    opti.subject_to(progress[0] == 0)
    opti.subject_to(progress[-1] == 1)
    opti.set_initial(progress, np.linspace(0, 1, nodes + 1))
    for k in range(1, nodes):
        state = transcription.X[k]
        opti.subject_to(state[0] == x_ref(length * progress[k]))
        opti.subject_to(state[1] == y_ref(length * progress[k]))
    # Collocation points alone can miss deviations within a coarse interval.
    samples = sorted(set((*transcription.collocation_roots, 0.25, 0.5, 0.75)))
    for k in range(nodes):
        previous = progress[k]
        for tau in samples:
            position = opti.variable()
            opti.subject_to(opti.bounded(0, position, 1))
            opti.subject_to(position >= previous)
            opti.set_initial(position, (k + tau) / nodes)
            state = transcription.state_at(k, tau)
            dx = (state[0] - x_ref(length * position)) / tolerance_m
            dy = (state[1] - y_ref(length * position)) / tolerance_m
            opti.subject_to(dx**2 + dy**2 <= 1)
            previous = position
        opti.subject_to(progress[k + 1] >= previous)
