from __future__ import annotations

import warnings
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import casadi as ca
import openap
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

try:
    from . import tools
except ImportError:
    warnings.warn("cfgrib and sklearn are required for wind integration")

from . import _dynamics, _objectives, _trajectory
from ._performance import build_performance_models
from ._types import LatLon

if TYPE_CHECKING:
    from ._options import TrajectoryResult


class Base:
    BADA4_MIN_FUELFLOW_KG_S: ClassVar[float] = 0.05
    MASS_CONSTRAINT_TOL_KG: ClassVar[float] = 1e-3
    VARIABLE_TIMESTEP_MIN_FACTOR: ClassVar[float] = 0.65
    VARIABLE_TIMESTEP_MAX_FACTOR: ClassVar[float] = 1.65

    # Attributes set by subclass init_conditions — declared here for pyright.
    # Runtime values are always assigned before _build_opti is called.
    x_lb: list
    x_ub: list
    x_0_lb: list
    x_0_ub: list
    x_f_lb: list
    x_f_ub: list
    u_lb: list
    u_ub: list
    u_0_lb: list
    u_0_ub: list
    u_f_lb: list
    u_f_ub: list
    x_guess: "np.ndarray"
    u_guess: list

    # Typed as Any so post-solve accesses (.stats(), .value()) don't need guards.
    # _last_solution is None before solve() and an OptiSol/OptiDebug after.
    _last_solution: Any = None

    def __init__(
        self,
        actype: str,
        origin: str | LatLon,
        destination: str | LatLon,
        m0: float = 0.85,
        engine: str | None = None,
        use_synonym: bool = False,
        dT: float = 0.0,
        performance_model: str = "openap",
        bada_path: str | None = None,
        payload: float | None = None,
    ) -> None:
        """OpenAP trajectory optimizer.

        Args:
            actype (str): ICAO aircraft type code
            origin (str | LatLon): ICAO or IATA code of airport, or tuple (lat, lon)
            destination (str | LatLon): ICAO or IATA code of airport, or tuple
                (lat, lon)
            m0 (float, optional): Takeoff mass factor. Defaults to 0.85 (of MTOW).
            engine (str | None, optional): Engine type. Defaults to aircraft's default
                engine.
            use_synonym (bool, optional): Use aircraft type synonym. Defaults to False.
            dT (float, optional): Temperature shift from standard ISA. Default = 0.
            performance_model: Performance model name: "openap", "bada3", or "bada4".
            bada_path: Path to BADA data when using BADA performance models.
            payload: Payload mass in kg. When provided, initial mass is optimized
                between OEW + payload and the aircraft's fuel/MTOW limit; m0 is used
                as the initial mass guess.
        """
        if isinstance(origin, str):
            ap1 = openap.nav.airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]  # type: ignore[index]  # openap stubs lack dict return type
        else:
            self.lat1, self.lon1 = origin

        if isinstance(destination, str):
            ap2 = openap.nav.airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]  # type: ignore[index]  # openap stubs lack dict return type
        else:
            self.lat2, self.lon2 = destination

        self.actype = actype
        self.performance_model = performance_model.lower()
        self.bada_path = bada_path
        self.use_synonym = use_synonym
        models = build_performance_models(
            self.actype,
            engine=engine,
            use_synonym=self.use_synonym,
            performance_model=self.performance_model,
            bada_path=self.bada_path,
        )
        self.performance_model = models.name
        self.aircraft = models.aircraft
        self.engtype = models.engtype
        self.engine = models.engine

        self.oew = self.aircraft["oew"]
        self.mlw = self.aircraft["mlw"]
        self.fuel_max = self.aircraft["mfc"]
        self.payload = payload
        self.mass_min = self._mass_min(payload)
        self.mass_init_lb, self.mass_init_ub = self._mass_init_bounds(m0)
        self.mass_init = min(
            max(m0 * self.aircraft["mtow"], self.mass_init_lb), self.mass_init_ub
        )
        if payload is not None:
            warnings.warn(
                "payload is provided; m0 is used only as the initial mass guess "
                "and does not fix the initial mass.",
                stacklevel=2,
            )
        self.mach_max = self.aircraft["mmo"]
        self.dT = dT

        native_actype = actype.split("-", 1)[0]
        self.thrust = models.thrust
        self.wrap = openap.WRAP(native_actype, use_synonym=self.use_synonym)
        self.drag = models.drag
        self.fuelflow = models.fuelflow
        self.emission = models.emission

        self.wind = None

        # Check cruise range
        self.range = oc.geo.distance(self.lat1, self.lon1, self.lat2, self.lon2)
        max_range = self.wrap.cruise_range()["maximum"] * 1.2
        if self.range > max_range * 1000:
            warnings.warn("The destination is likely out of maximum cruise range.")

        self.debug = False
        self._last_solution = None
        self.objective_value: float | None = None
        self.setup()

    def _mass_min(self, payload: float | None) -> float:
        """Minimum physically allowed mass for the optimized trajectory."""
        if payload is None:
            return self.oew
        if payload < 0:
            raise ValueError("payload must be non-negative")

        mass_min = self.oew + payload
        if mass_min > self.aircraft["mtow"]:
            raise ValueError("OEW + payload must not exceed MTOW")
        return mass_min

    def _mass_init_bounds(self, m0: float) -> tuple[float, float]:
        """Initial-mass bounds; fixed unless payload optimization is enabled."""
        mass_guess = m0 * self.aircraft["mtow"]
        if self.payload is None:
            return mass_guess, mass_guess

        mass_init_ub = min(self.aircraft["mtow"], self.mass_min + self.fuel_max)
        if mass_init_ub <= self.mass_min:
            raise ValueError("payload leaves no feasible fuel capacity")
        return self.mass_min, mass_init_ub

    @property
    def solver(self):
        """Deprecated: use ``optimizer.stats`` or ``optimizer.success``."""
        warnings.warn(
            "optimizer.solver is deprecated; use optimizer.stats or optimizer.success. "
            "Will be removed in v2.3.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._last_solution

    @property
    def stats(self) -> dict:
        """Solver stats dict from the most recent solve."""
        if self._last_solution is None:
            raise RuntimeError(
                "No solver stats available — call trajectory() or"
                " multi_start_trajectory() first."
            )
        return self._last_solution.stats()

    @property
    def success(self) -> bool:
        """Whether the most recent solve succeeded."""
        if self._last_solution is None:
            raise RuntimeError(
                "No solver result available — call trajectory() or"
                " multi_start_trajectory() first."
            )
        return bool(self._last_solution.stats()["success"])

    def proj(
        self, lon: Any, lat: Any, inverse: bool = False, symbolic: bool = False
    ) -> Any:
        """Project between lon/lat and local cartesian coordinates.

        Uses azimuthal equidistant projection centered between origin
        and destination.

        Args:
            lon: Longitude (forward) or x coordinate (inverse).
            lat: Latitude (forward) or y coordinate (inverse).
            inverse: If True, convert (x, y) back to (lon, lat).
            symbolic: If True, use CasADi symbolic math.

        Returns:
            tuple: (x, y) in meters, or (lon, lat) if inverse.
        """
        lat0 = (self.lat1 + self.lat2) / 2
        lon0 = (self.lon1 + self.lon2) / 2

        if symbolic:
            geo, trig = oc.geo, ca
        else:
            geo, trig = openap.aero, np

        if not inverse:
            bearings = geo.bearing(lat0, lon0, lat, lon) / 180 * np.pi
            distances = geo.distance(lat0, lon0, lat, lon)
            return distances * trig.sin(bearings), distances * trig.cos(bearings)
        else:
            x, y = lon, lat
            distances = trig.sqrt(x**2 + y**2)
            bearing = trig.arctan2(x, y) * 180 / np.pi
            lat, lon = geo.latlon(lat0, lon0, distances, bearing)
            return lon, lat

    def _compute_bbox(
        self, margin_m: float = 10_000, waypoints: Any = None
    ) -> tuple[float, float, float, float]:
        """Compute projected bounding box around origin/destination with margin.

        Returns (x_min, x_max, y_min, y_max) in projected meters.
        """
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        xs = [xp_0, xp_f]
        ys = [yp_0, yp_f]
        for lat, lon in self._normalize_waypoints(waypoints):
            xp, yp = self.proj(lon, lat)
            xs.append(xp)
            ys.append(yp)

        x_min = min(xs) - margin_m
        x_max = max(xs) + margin_m
        y_min = min(ys) - margin_m
        y_max = max(ys) + margin_m
        return x_min, x_max, y_min, y_max

    def _compute_bearing_psi(self) -> float:
        """Great-circle bearing from origin to destination, in radians."""
        hdg = oc.geo.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        return hdg * np.pi / 180

    def initial_guess(self, flight: pd.DataFrame | None = None) -> np.ndarray:
        """Generate initial guess for the optimizer.

        Args:
            flight: Existing trajectory to use as initial guess.
                If None, uses straight-line interpolation at cruise altitude.

        Returns:
            np.ndarray: Array of shape (nodes+1, 5) with columns
                [xp, yp, h, mass, ts].
        """
        return _dynamics.great_circle_init(
            self.lat1,
            self.lon1,
            self.lat2,
            self.lon2,
            n_nodes=self.nodes,
            mass_init=self.mass_init,
            aircraft=self.aircraft,
            proj=self.proj,
            flight=flight,
        )

    def enable_wind(self, windfield: pd.DataFrame) -> None:
        """Enable wind field integration using polynomial regression model.

        Args:
            windfield: DataFrame with columns [longitude, latitude, h, ts, u, v].
        """
        self.wind = tools.PolyWind(
            windfield, self.proj, self.lat1, self.lon1, self.lat2, self.lon2
        )

    def collocation_coeff(self):
        """Compute Legendre collocation coefficients.

        Returns:
            tuple: (C, D, B) where C is the derivative matrix,
                D is the continuity vector, B is the quadrature vector.
        """
        return _dynamics.collocation_coeff(self.polydeg)

    def xdot(self, x, u) -> ca.MX:
        """State derivatives for the equations of motion.

        Args:
            x: State vector [xp (m), yp (m), h (m), mass (kg), ts (s)].
            u: Control vector [mach, vs (m/s), heading (rad)].

        Returns:
            ca.MX: State derivatives [dx, dy, dh, dm, dt].
        """
        return _dynamics.xdot(
            x,
            u,
            fuelflow=self.fuelflow,
            dT=self.dT,
            wind=self.wind,
        )

    def setup(
        self,
        nodes: int | None = None,
        polydeg: int = 3,
        debug: bool = False,
        max_nodes: int = 120,
        max_iter: int = 3000,
        tol: float = 1e-6,
        acceptable_tol: float = 1e-4,
        ipopt_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Configure the optimizer discretization and solver settings.

        Args:
            nodes: Number of collocation intervals. Explicit values are honored.
                The default is auto-computed from distance (~1 per 50 km,
                clamped to [20, max_nodes]).
            polydeg: Collocation polynomial degree. Default 3 (Legendre).
            debug: Print solver output. Default False.
            max_nodes: Upper limit for auto-computed nodes. Default 120.
            max_iter: IPOPT maximum iterations. Default 3000.
            tol: IPOPT convergence tolerance. Default 1e-6.
            acceptable_tol: IPOPT acceptable tolerance. Default 1e-4.
            ipopt_kwargs: Additional IPOPT options as {key: value}.
                Keys are passed as "ipopt.{key}".
        """
        if ipopt_kwargs is None:
            ipopt_kwargs = {}
        if nodes is not None:
            if nodes < 1:
                raise ValueError("nodes must be positive")
            self.nodes = nodes
        else:
            self.nodes = int(self.range / 50_000)  # node every 50km
            self.nodes = max(20, self.nodes)
            self.nodes = min(max_nodes, self.nodes)

        self.polydeg = polydeg

        self.debug = debug

        if debug:
            print("Calculating optimal trajectory...")
            ipopt_print = 5
            print_time = 1
        else:
            ipopt_print = 0
            print_time = 0

        self.solver_options = {
            # Convert Opti bounded() constraints to IPOPT variable bounds (lbx/ubx)
            "detect_simple_bounds": True,
            "print_time": print_time,
            "calc_lam_p": False,
            "ipopt.print_level": ipopt_print,
            "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter,
            "ipopt.fixed_variable_treatment": "relax_bounds",
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": acceptable_tol,
            "ipopt.mu_strategy": "adaptive",
        }

        for key, value in ipopt_kwargs.items():
            self.solver_options[f"ipopt.{key}"] = value

    def init_model(self, objective, **kwargs):
        """Build the symbolic dynamics function for the given objective.

        Creates self.x (states), self.u (controls), and self.func_dynamics.
        Must be called after self.ts_final and self.dt are set.

        Args:
            objective: Objective name (str), "ci:N" format, or callable(x, u, dt).
            **kwargs: Passed to the objective function.
        """
        # Model variables — CasADi stubs incorrectly type MX.sym(str) as MX.sym(MX)
        xp = ca.MX.sym("xp")  # type: ignore[arg-type]
        yp = ca.MX.sym("yp")  # type: ignore[arg-type]
        h = ca.MX.sym("h")  # type: ignore[arg-type]
        m = ca.MX.sym("m")  # type: ignore[arg-type]
        ts = ca.MX.sym("ts")  # type: ignore[arg-type]

        mach = ca.MX.sym("mach")  # type: ignore[arg-type]
        vs = ca.MX.sym("vs")  # type: ignore[arg-type]
        psi = ca.MX.sym("psi")  # type: ignore[arg-type]

        self.x = ca.vertcat(xp, yp, h, m, ts)
        self.u = ca.vertcat(mach, vs, psi)

        interval_dt = ca.MX.sym("dt")  # type: ignore[arg-type]
        # self.ts_final and self.dt are set by _build_opti() before this call

        # Handle objective function. User callables expect `self.obj_*`
        # helpers (see tests/test_full_flight.py), so they run as-is. String
        # specs go through the pure-function registry with model context
        # injected here.
        if isinstance(objective, Callable):
            self.objective = objective
            L = self.objective(self.x, self.u, interval_dt, **kwargs)
        else:
            resolved = _objectives.resolve_objective(objective)
            ctx = self._objective_ctx()
            ctx.update(kwargs)
            self.objective = lambda x, u, dt, **kw: resolved(x, u, dt, **{**ctx, **kw})
            L = resolved(self.x, self.u, interval_dt, **ctx)

        # Continuous time dynamics
        self.func_dynamics = ca.Function(
            "f",
            [self.x, self.u, interval_dt],
            [self.xdot(self.x, self.u), L],
            ["x", "u", "dt"],
            ["xdot", "L"],
            {"allow_free": True},
        )

    def _build_opti(self, objective, ts_final_guess, **kwargs):
        """Build CasADi Opti problem with direct collocation structure.

        Creates the Opti instance, free final time variable, calls init_model,
        and builds the collocation equations, variable bounds, and initial guesses.

        Must be called after init_conditions() which sets the bound attributes.

        Args:
            objective: Objective function name or callable.
            ts_final_guess: Initial guess for total flight time (seconds).
            **kwargs: Solver construction options and objective options.

        Returns:
            tuple: (X, U) where X is list of state MX vars at each node boundary
                   (length nodes+1), U is list of control MX vars (length nodes).
        """
        self._opti = ca.Opti()

        # Grid-cost interpolants (bspline) need IPOPT's exact Hessian for
        # numerical stability. For string objective specs this is driven by
        # the registry's `requires_exact_hessian` flag; for user-supplied
        # callables (which might internally call obj_grid_cost) the same
        # effect is triggered by the presence of an `interpolant` kwarg.
        needs_exact_hessian = False
        if not isinstance(objective, Callable):
            try:
                resolved = _objectives.resolve_objective(objective)
                needs_exact_hessian = getattr(resolved, "requires_exact_hessian", False)
            except (ValueError, TypeError):
                # Defer the error until init_model so the message matches the
                # existing dispatch path.
                pass
        if kwargs.get("interpolant") is not None:
            needs_exact_hessian = True
        if kwargs.get("exact_hessian", False):
            needs_exact_hessian = True
        if needs_exact_hessian:
            self.solver_options["ipopt.hessian_approximation"] = "exact"

        # Free final time — must be set before init_model
        self.ts_final = self._opti.variable()
        self._opti.subject_to(self.ts_final >= 0)
        self._opti.set_initial(self.ts_final, ts_final_guess)
        self.dt = self.ts_final / self.nodes
        variable_timestep = bool(
            kwargs.get("variable_timestep", kwargs.get("waypoints") is not None)
        )
        self._variable_timestep = variable_timestep
        self._interval_dts: list[Any] = []
        model_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "waypoints",
                "waypoint_tolerance_m",
                "waypoint_node_indices",
                "variable_timestep",
                "dt_min",
                "dt_max",
            }
        }

        # Build dynamics function (captures self.dt with free ts_final)
        self.init_model(objective, **model_kwargs)

        C, D, B = self.collocation_coeff()
        nstates = self.x.shape[0]

        X = []  # States at node boundaries (length: nodes + 1)
        U = []  # Controls at each node (length: nodes)
        J = 0  # Objective accumulator

        # Initial state
        Xk = self._opti.variable(nstates)
        self._opti.subject_to(self._opti.bounded(self.x_0_lb, Xk, self.x_0_ub))  # type: ignore[arg-type]  # CasADi stubs wrong: bounded(lb, expr, ub) accepts lists
        self._opti.set_initial(Xk, self.x_guess[0])
        X.append(Xk)

        for k in range(self.nodes):
            if variable_timestep:
                interval_dt = self._opti.variable()
                dt_min, dt_max, interval_guess = self._variable_timestep_bounds(
                    ts_final_guess,
                    dt_min=kwargs.get("dt_min"),
                    dt_max=kwargs.get("dt_max"),
                )
                self._opti.subject_to(self._opti.bounded(dt_min, interval_dt, dt_max))  # type: ignore[arg-type]  # CasADi stubs wrong
                self._opti.set_initial(interval_dt, interval_guess)
                self._interval_dts.append(interval_dt)
            else:
                interval_dt = self.dt

            # Control variable
            Uk = self._opti.variable(self.u.shape[0])
            U.append(Uk)

            if k == 0:
                u_lb, u_ub = self.u_0_lb, self.u_0_ub
            elif k == self.nodes - 1:
                u_lb, u_ub = self.u_f_lb, self.u_f_ub
            else:
                u_lb, u_ub = self.u_lb, self.u_ub

            self._opti.subject_to(self._opti.bounded(u_lb, Uk, u_ub))  # type: ignore[arg-type]  # CasADi stubs wrong
            self._opti.set_initial(Uk, self.u_guess)

            # Collocation points within this interval
            Xc = []
            for j in range(self.polydeg):
                Xkj = self._opti.variable(nstates)
                Xc.append(Xkj)
                self._opti.subject_to(self._opti.bounded(self.x_lb, Xkj, self.x_ub))  # type: ignore[arg-type]  # CasADi stubs wrong
                self._opti.set_initial(Xkj, self.x_guess[k])

            # Collocation equations and quadrature
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                fj, qj = self.func_dynamics(Xc[j - 1], Uk, interval_dt)  # type: ignore[misc]  # CasADi Function.__call__ return is opaque to pyright
                self._constrain_performance_model_domain(fj)
                self._opti.subject_to(interval_dt * fj == xpc)

                Xk_end = Xk_end + D[j] * Xc[j - 1]
                J = J + B[j] * qj

            # State at end of interval
            Xk = self._opti.variable(nstates)
            X.append(Xk)

            if k < self.nodes - 1:
                x_lb, x_ub = self.x_lb, self.x_ub
            else:
                x_lb, x_ub = self.x_f_lb, self.x_f_ub

            self._opti.subject_to(self._opti.bounded(x_lb, Xk, x_ub))  # type: ignore[arg-type]  # CasADi stubs wrong
            self._opti.set_initial(Xk, self.x_guess[k])

            # Continuity constraint
            self._opti.subject_to(Xk_end == Xk)

        if variable_timestep:
            self._opti.subject_to(
                ca.sum1(ca.vertcat(*self._interval_dts)) == self.ts_final
            )

        # Optional: rescale the objective by its value at the initial guess
        # so IPOPT sees f(x0) ≈ 1. Important for objectives whose natural
        # magnitude is far from O(1) — for example climate-metric objectives
        # combining contrail ATR20 (~1e-12 K/s) and CO2 (~7e-15 K/kg·s),
        # where IPOPT's default termination tolerances would otherwise be
        # satisfied at any feasible point. The physical objective_value is
        # multiplied back in _solve so callers see the unscaled number.
        self._objective_rescale = 1.0
        if kwargs.get("auto_rescale_objective", False):
            x_init = self._opti.debug.value(self._opti.x, self._opti.initial())
            j_at_init = ca.Function("j_at_init", [self._opti.x], [J])
            f0 = float(j_at_init(x_init))  # type: ignore[arg-type]  # CasADi Function.__call__ return type is opaque to pyright
            # Only skip rescaling if f0 is essentially zero to avoid
            # divide-by-zero. Otherwise rescale by abs(f0) in either
            # direction — crucial for climate-metric objectives where
            # the natural magnitude is far below 1.
            if abs(f0) > 1e-30:
                self._objective_rescale = abs(f0)
                J = J / self._objective_rescale

        self._opti.minimize(J)

        return X, U

    def _interval_dt(self, k: int) -> Any:
        if getattr(self, "_variable_timestep", False):
            return self._interval_dts[k]
        return self.dt

    def _variable_timestep_bounds(
        self,
        ts_final_guess: float,
        *,
        dt_min: float | None = None,
        dt_max: float | None = None,
    ) -> tuple[float, float, float]:
        interval_guess = ts_final_guess / self.nodes
        if dt_min is None:
            dt_min = max(5.0, self.VARIABLE_TIMESTEP_MIN_FACTOR * interval_guess)
        if dt_max is None:
            dt_max = min(
                self.x_f_ub[4],
                max(dt_min, self.VARIABLE_TIMESTEP_MAX_FACTOR * interval_guess),
            )
        if not (np.isfinite(dt_min) and np.isfinite(dt_max) and 0 < dt_min <= dt_max):
            raise ValueError(
                "variable timestep bounds must be finite, positive, and ordered "
                "with dt_min <= dt_max"
            )
        return dt_min, dt_max, min(max(interval_guess, dt_min), dt_max)

    def _normalize_waypoints(self, waypoints: Any = None) -> list[LatLon]:
        if waypoints is None:
            return []
        normalized = []
        for waypoint in waypoints:
            if len(waypoint) != 2:
                raise ValueError("waypoints must be (lat, lon) pairs")
            lat, lon = float(waypoint[0]), float(waypoint[1])
            if not -90 <= lat <= 90:
                raise ValueError(f"waypoint latitude out of range: {lat}")
            if not -180 <= lon <= 180:
                raise ValueError(f"waypoint longitude out of range: {lon}")
            normalized.append((lat, lon))
        return normalized

    def _waypoint_node_indices(
        self, waypoints: list[LatLon], waypoint_node_indices: Any = None
    ) -> list[int]:
        if not waypoints:
            return []
        if len(waypoints) >= self.nodes:
            raise ValueError("number of waypoints must be smaller than optimizer nodes")

        if waypoint_node_indices is None:
            xp_0, yp_0 = self.proj(self.lon1, self.lat1)
            xp_f, yp_f = self.proj(self.lon2, self.lat2)
            points = [(xp_0, yp_0)]
            points.extend(self.proj(lon, lat) for lat, lon in waypoints)
            points.append((xp_f, yp_f))
            segment_lengths = [
                float(np.hypot(x_b - x_a, y_b - y_a))
                for (x_a, y_a), (x_b, y_b) in pairwise(points)
            ]
            total_length = sum(segment_lengths)
            if total_length <= 0:
                return list(range(1, len(waypoints) + 1))

            indices = []
            cumulative = 0.0
            for i, segment_length in enumerate(segment_lengths[:-1]):
                cumulative += segment_length
                raw_index = round(cumulative / total_length * self.nodes)
                lower = indices[-1] + 1 if indices else 1
                upper = self.nodes - (len(waypoints) - i)
                indices.append(max(lower, min(upper, raw_index)))
            return indices

        indices = [int(index) for index in waypoint_node_indices]
        if len(indices) != len(waypoints):
            raise ValueError("waypoint_node_indices must match number of waypoints")
        if any(index <= 0 or index >= self.nodes for index in indices):
            raise ValueError("waypoint_node_indices must be interior node indices")
        if any(b <= a for a, b in pairwise(indices)):
            raise ValueError("waypoint_node_indices must be strictly increasing")
        return indices

    def _constrain_waypoints(
        self,
        X: list[Any],
        waypoints: Any = None,
        *,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_node_indices: Any = None,
    ) -> None:
        normalized = self._normalize_waypoints(waypoints)
        if not normalized:
            return
        if waypoint_tolerance_m <= 0:
            raise ValueError("waypoint_tolerance_m must be positive")

        indices = self._waypoint_node_indices(normalized, waypoint_node_indices)
        for index, (lat, lon) in zip(indices, normalized):
            xp, yp = self.proj(lon, lat)
            dist2 = (X[index][0] - xp) ** 2 + (X[index][1] - yp) ** 2
            self._opti.subject_to(dist2 <= waypoint_tolerance_m**2)

    def _constrain_performance_model_domain(self, xdot: ca.MX) -> None:
        """Keep symbolic dynamics inside model-valid regions.

        BADA4 fuel-flow polynomials can become negative at low-Mach/high-altitude
        states. The mass derivative is ``-fuel_flow`` in kg/s, so constraining
        ``-xdot[3]`` at collocation points prevents IPOPT from exploiting those
        nonphysical regions in both dynamics and objective integration.
        """
        if self.performance_model == "bada4":
            self._opti.subject_to(-xdot[3] >= self.BADA4_MIN_FUELFLOW_KG_S)

    def _solve(self, X, U, **kwargs):
        """Solve the Opti NLP and extract trajectory DataFrame.

        Args:
            X: List of state MX variables from _build_opti.
            U: List of control MX variables from _build_opti.
            **kwargs: Passed through to to_trajectory().

        Returns:
            pd.DataFrame: Trajectory DataFrame.
        """
        self._opti.solver("ipopt", self.solver_options)

        try:
            sol = self._opti.solve()
        except RuntimeError as e:
            if self.debug:
                warnings.warn(f"Solver failed: {e}")
            sol = self._opti.debug

        self._last_solution = sol
        # Undo auto_rescale_objective so callers always see the physical value.
        self.objective_value = float(sol.value(self._opti.f)) * self._objective_rescale

        ts_final_val = float(sol.value(self.ts_final))
        x_opt = sol.value(ca.horzcat(*X))
        u_opt = sol.value(ca.horzcat(*U))

        return self.to_trajectory(ts_final_val, x_opt, u_opt, **kwargs)

    def _thrust_climb(self, tas: Any, alt: Any) -> Any:
        if self.performance_model == "openap":
            return self.thrust.climb(tas, alt, 0, dT=self.dT)
        return self.thrust.climb(tas, alt, dT=self.dT)

    def _clean_drag_polar_params(self, tas: Any, alt: Any) -> dict[str, Any]:
        if hasattr(self.drag, "clean_drag_polar_params"):
            return self.drag.clean_drag_polar_params(tas=tas, alt=alt, dT=self.dT)
        polar = self.drag.polar["clean"]
        return {"cd0": polar["cd0"], "cd2": polar["k"], "cd6": 0.0}

    def _lift_margin_drag(
        self, mass: Any, tas: Any, alt: Any, margin: float = 0.8
    ) -> Any:
        h = alt * ft
        v = tas * kts
        area = self.drag.S if hasattr(self.drag, "S") else self.aircraft["wing"]["area"]
        rho = oc.aero.density(h, dT=self.dT)
        qS = 0.5 * rho * v**2 * area
        cl_margin = mass * oc.aero.g0 / (qS * margin + 1e-10)
        params = self._clean_drag_polar_params(tas, alt)
        cd = (
            params["cd0"]
            + params["cd2"] * cl_margin**2
            + params.get("cd6", 0.0) * cl_margin**6
        )
        return cd * qS

    def _constrain_clean_performance(
        self,
        opti: ca.Opti,
        mass: Any,
        tas: Any,
        alt: Any,
        thrust_max: Any,
        *,
        drag_margin: float = 0.95,
        lift_drag_margin: float = 0.9,
    ) -> Any:
        drag = self.drag.clean(mass, tas, alt, dT=self.dT)
        opti.subject_to(thrust_max * drag_margin >= drag)
        opti.subject_to(
            thrust_max * lift_drag_margin >= self._lift_margin_drag(mass, tas, alt)
        )
        return drag

    def _calc_emission(self, x, u, symbolic=True):
        """Compute emission species from state and control vectors.

        Returns:
            tuple: (co2, h2o, sox, soot, nox) emission rates.
        """
        _, _, h, m = x[0], x[1], x[2], x[3]
        mach, vs, _ = u[0], u[1], u[2]

        if symbolic:
            fuelflow = self.fuelflow
            emission = self.emission
            v = oc.aero.mach2tas(mach, h, dT=self.dT)
        else:
            fuelflow = openap.FuelFlow(
                self.actype, self.engtype, polydeg=2, use_synonym=self.use_synonym
            )
            emission = openap.Emission(
                self.actype, self.engtype, use_synonym=self.use_synonym
            )
            v = openap.aero.mach2tas(mach, h, dT=self.dT)  # type: ignore[arg-type]  # openap stubs type dT as int; float is correct

        ff = fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        co2 = emission.co2(ff)
        h2o = emission.h2o(ff)
        sox = emission.sox(ff)
        soot = emission.soot(ff)
        nox = emission.nox(ff, v / kts, h / ft, dT=self.dT)

        return co2, h2o, sox, soot, nox

    # Preserved for back-compat on `Base._CLIMATE_COEFF`; authoritative copy
    # lives in opentop._objectives.
    _CLIMATE_COEFF: ClassVar[dict] = _objectives._CLIMATE_COEFF

    def _objective_ctx(self):
        """Build the model context dict pure objectives need as kwargs."""
        return {
            "fuelflow": self.fuelflow,
            "dT": self.dT,
            "actype": self.actype,
            "engtype": self.engtype,
            "use_synonym": self.use_synonym,
            "performance_model": self.performance_model,
            "bada_path": self.bada_path,
            "proj": self.proj,
            "calc_emission": self._calc_emission,
        }

    def obj_fuel(
        self, x: ca.MX, u: ca.MX, dt: ca.MX, symbolic: bool = True, **kwargs: Any
    ) -> ca.MX:
        """Fuel burn objective: fuelflow * dt. Delegates to _objectives."""
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_fuel(x, u, dt, symbolic=symbolic, **ctx)

    def obj_time(self, x: ca.MX, u: ca.MX, dt: ca.MX, **kwargs: Any) -> ca.MX:
        """Minimum time objective. Delegates to _objectives."""
        return _objectives.obj_time(x, u, dt, **kwargs)

    def obj_ci(
        self,
        x: ca.MX,
        u: ca.MX,
        dt: ca.MX,
        ci: float,
        time_price: float = 25,
        fuel_price: float = 0.8,
        **kwargs: Any,
    ) -> ca.MX:
        """Cost index objective blending time and fuel costs.

        Args:
            ci: Cost index (0-100). 0 = fuel only, 100 = time only.
            time_price: Cost of time in EUR/min. Default 25.
            fuel_price: Cost of fuel in EUR/L. Default 0.8.
        """
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_ci(
            x,
            u,
            dt,
            ci=ci,
            time_price=time_price,
            fuel_price=fuel_price,
            **ctx,
        )

    def _obj_climate(
        self, x: ca.MX, u: ca.MX, dt: ca.MX, metric: str, **kwargs: Any
    ) -> ca.MX:
        """Climate impact objective using GWP/GTP metric coefficients."""
        ctx = self._objective_ctx()
        ctx.update(kwargs)
        return _objectives.obj_climate(x, u, dt, metric=metric, **ctx)

    def obj_grid_cost(self, x: ca.MX, u: ca.MX, dt: ca.MX, **kwargs: Any) -> ca.MX:
        """Grid-based cost objective using a CasADi interpolant.

        Args:
            **kwargs:
                interpolant: CasADi interpolant function.
                symbolic: Use symbolic computation. Default True.
                n_dim: Input dimension, 3 (lon,lat,h) or 4 (+ts).
                    Auto-detected from the interpolant by default.
                time_dependent: Multiply cost by dt. Default True.
        """
        return _objectives.obj_grid_cost(x, u, dt, proj=self.proj, **kwargs)

    def to_trajectory(
        self, ts_final: float, x_opt: np.ndarray, u_opt: np.ndarray, **kwargs: Any
    ) -> pd.DataFrame:
        """Convert optimization results to a trajectory DataFrame.

        Args:
            ts_final: Final timestamp
            x_opt: Optimized states
            u_opt: Optimized controls
            **kwargs: Additional arguments including:
                - interpolant: Grid cost interpolant function
                - time_dependent: Whether grid cost is time dependent (default True)
                - n_dim: Dimension of grid cost, 3 or 4 (auto-detected from
                  interpolant if not provided)

        Returns:
            pd.DataFrame: Trajectory with columns including fuel_cost and grid_cost
        """
        # Historically to_trajectory accepted arbitrary kwargs (via
        # kwargs.get) and silently ignored unknowns; preserve that contract
        # by passing only the recognised ones through to the pure helper.
        df, X, U, dt = _trajectory.to_dataframe(
            ts_final,
            x_opt,
            u_opt,
            proj=self.proj,
            nodes=self.nodes,
            dT=self.dT,
            wind=self.wind,
            actype=self.actype,
            engtype=self.engtype,
            use_synonym=self.use_synonym,
            performance_model=self.performance_model,
            bada_path=self.bada_path,
            interpolant=kwargs.get("interpolant", None),
            time_dependent=kwargs.get("time_dependent", True),
            n_dim=kwargs.get("n_dim"),
        )
        # Preserve historical side effects for downstream callers/tests.
        self.X = X
        self.U = U
        self.dt = dt
        return df

    def _make_result(self, df: pd.DataFrame | None) -> TrajectoryResult:
        """Package a trajectory DataFrame into a TrajectoryResult.

        Used when trajectory(result_object=True). ``df`` may be None (from a
        rejected solve); it is coerced to an empty DataFrame in the result.
        """
        from ._options import build_result

        stats = (
            dict(self._last_solution.stats())
            if getattr(self, "_last_solution", None)
            else {}
        )
        obj = (
            float(self.objective_value)
            if self.objective_value is not None
            else float("nan")
        )
        return build_result(df, stats, obj)

    def multi_start_trajectory(
        self,
        objective: str | Callable = "fuel",
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, list[dict]]:
        from . import _multi_start

        return _multi_start.run_multi_start(self, objective, **kwargs)
