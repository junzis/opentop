"""CompleteFlight — full-flight (takeoff to landing) trajectory optimizer."""

from __future__ import annotations

import warnings
from math import pi
from typing import TYPE_CHECKING, Any, Callable

import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from ._types import LatLon
from .base import Base

if TYPE_CHECKING:
    from ._options import TrajectoryResult


class CompleteFlight(Base):
    """Complete flight (takeoff to landing) trajectory optimizer."""

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
        super().__init__(
            actype,
            origin,
            destination,
            m0=m0,
            engine=engine,
            use_synonym=use_synonym,
            dT=dT,
            performance_model=performance_model,
            bada_path=bada_path,
            payload=payload,
        )

    def init_conditions(self, **kwargs: Any) -> None:
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to Cartesian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)
        x_min, x_max, y_min, y_max = self._compute_bbox(
            waypoints=kwargs.get("waypoints")
        )

        ts_min = 0
        ts_max = max(5, self.range / 1000 / 500) * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = 100 * ft

        psi = self._compute_bearing_psi()
        min_mach = 0.3 if self.performance_model == "bada4" else 0.1
        mass_lower = self.mass_min if self.payload is not None else self.oew * 0.5

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, self.mass_init_lb, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_min, self.mass_init_ub, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_lower, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, self.mass_init_ub, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_lower, ts_min]
        self.x_ub = [x_max, y_max, h_max, self.mass_init_ub, ts_max]

        # Control init - lower and upper bounds
        self.u_0_lb = [min_mach, 500 * fpm, psi]
        self.u_0_ub = [0.3, 2500 * fpm, psi]

        # Control final - lower and upper bounds
        self.u_f_lb = [min_mach, -1500 * fpm, psi]
        self.u_f_ub = [0.3, -300 * fpm, psi]

        # Control - Lower and upper bound
        self.u_lb = [min_mach, -2500 * fpm, psi - pi / 2]
        self.u_ub = [self.mach_max, 2500 * fpm, psi + pi / 2]

        # Initial guess for the states
        self.x_guess = self.initial_guess()
        phase_indices = self._phase_node_indices(
            climb_nodes=kwargs.get("climb_nodes"),
            descent_nodes=kwargs.get("descent_nodes"),
        )
        if phase_indices is not None:
            self._apply_phase_altitude_guess(phase_indices, h_min, h_max)

        # Control - guesses
        self.u_guess = [0.6, 1000 * fpm, psi]

    def _phase_node_indices(
        self,
        *,
        climb_nodes: int | None = None,
        descent_nodes: int | None = None,
    ) -> tuple[int, int]:
        """Return the top-of-climb and top-of-descent node indices."""
        if self.nodes < 3:
            raise ValueError(
                "complete flights need at least three intervals for climb, "
                "cruise, and descent"
            )

        # Fixed 500/300 km phase lengths can consume an entire short route;
        # thirds also give coarse meshes enough nodes in every phase.
        if self.nodes <= 40 or self.range <= 800_000:
            default_climb = default_descent = max(3, self.nodes // 3)
        else:
            interval_distance = self.range / (self.nodes + 1)
            default_climb = max(10, int(500_000 / interval_distance))
            default_descent = max(10, int(300_000 / interval_distance))

        # Automatic allocations must leave at least one interval for cruise.
        available_phase_nodes = self.nodes - 1
        if default_climb + default_descent > available_phase_nodes:
            climb_share = default_climb / (default_climb + default_descent)
            default_climb = round(available_phase_nodes * climb_share)
            default_climb = min(max(1, default_climb), available_phase_nodes - 1)
            default_descent = available_phase_nodes - default_climb

        if climb_nodes is not None:
            climb_nodes = int(climb_nodes)
            if climb_nodes < 1:
                raise ValueError("climb_nodes must be positive")
        if descent_nodes is not None:
            descent_nodes = int(descent_nodes)
            if descent_nodes < 1:
                raise ValueError("descent_nodes must be positive")

        if climb_nodes is None:
            descent = default_descent if descent_nodes is None else descent_nodes
            climb_nodes = min(default_climb, self.nodes - descent - 1)
        if descent_nodes is None:
            descent_nodes = min(default_descent, self.nodes - climb_nodes - 1)

        if climb_nodes < 1:
            raise ValueError("climb_nodes must be positive")
        if descent_nodes < 1:
            raise ValueError("descent_nodes must be positive")
        if climb_nodes + descent_nodes >= self.nodes:
            raise ValueError(
                "climb_nodes + descent_nodes must leave at least one cruise interval"
            )

        return climb_nodes, self.nodes - descent_nodes

    def _apply_phase_altitude_guess(
        self, phase_indices: tuple[int, int], h_min: float, h_max: float
    ) -> None:
        """Shape the complete-flight altitude guess around phase boundaries."""
        idx_toc, idx_tod = phase_indices
        h_cruise = min(self.aircraft["cruise"]["height"], h_max)
        h_guess = np.empty(self.nodes + 1)
        h_guess[: idx_toc + 1] = np.linspace(h_min, h_cruise, idx_toc + 1)
        h_guess[idx_toc : idx_tod + 1] = h_cruise
        h_guess[idx_tod:] = np.linspace(h_cruise, h_min, self.nodes - idx_tod + 1)
        self.x_guess[:, 2] = h_guess

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
        """Configure complete-flight discretization and solver settings."""
        if nodes is None:
            nodes = min(max_nodes, max(30, int(self.range / 50_000)))
        super().setup(
            nodes=nodes,
            polydeg=polydeg,
            debug=debug,
            max_nodes=max_nodes,
            max_iter=max_iter,
            tol=tol,
            acceptable_tol=acceptable_tol,
            ipopt_kwargs=ipopt_kwargs,
        )

    def _cruise_vertical_rate_limit(self) -> float:
        """Return cruise vertical-rate limit in m/s."""
        if self.performance_model == "bada4":
            return 100 * fpm
        return 500 * fpm

    def _cruise_mach_min(self) -> float | None:
        """Return model-specific minimum cruise Mach, if any."""
        if self.performance_model == "bada4":
            return 0.72
        return None

    def _mach_change_limit(self) -> float:
        """Return per-node Mach-change limit."""
        if self.performance_model == "bada4":
            return 0.08
        return 0.2

    def trajectory(
        self,
        objective: str | Callable = "fuel",
        *,
        max_fuel: float | None = None,
        return_failed: bool = False,
        initial_guess: pd.DataFrame | None = None,
        remove_cruise: bool = False,
        interpolant: Any = None,
        n_dim: int | None = None,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        waypoints: list[LatLon] | None = None,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_node_indices: list[int] | None = None,
        climb_nodes: int | None = None,
        descent_nodes: int | None = None,
        variable_timestep: bool | None = None,
        dt_min: float | None = None,
        dt_max: float | None = None,
        result_object: bool = False,
    ) -> pd.DataFrame | TrajectoryResult:
        """Compute the optimal complete flight trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            max_fuel: Maximum fuel constraint (kg).
            return_failed: Return result even if optimization fails.
            initial_guess: DataFrame to use as initial guess.
            remove_cruise: Unused for complete flight (accepted for API symmetry).
            interpolant: CasADi grid-cost interpolant (optional).
            n_dim: Interpolant input dimension (3 or 4). Auto-detected
                from the interpolant by default.
            time_dependent: Grid cost is time-dependent. Default False.
            auto_rescale_objective: Rescale objective to O(1). Default False.
            exact_hessian: Force IPOPT exact Hessian. Default False.
            waypoints: Ordered waypoint list as (lat, lon) pairs.
            waypoint_tolerance_m: Maximum waypoint miss distance in meters.
            waypoint_node_indices: Optional interior node indices assigned to
                waypoints. Defaults to evenly spaced ordered interior nodes.
            climb_nodes: Optional number of initial intervals reserved for climb.
            descent_nodes: Optional number of final intervals reserved for descent.
            variable_timestep: Optimize interval durations. Defaults to True
                when waypoints are supplied, otherwise False.
            dt_min: Minimum interval duration in seconds for variable timesteps.
                Defaults to an automatic fraction of the expected interval duration.
            dt_max: Maximum interval duration in seconds for variable timesteps.
            result_object: If True, return a TrajectoryResult.

        Returns:
            pd.DataFrame (or TrajectoryResult if result_object=True).
        """
        _kwargs = {
            "initial_guess": initial_guess,
            "interpolant": interpolant,
            "n_dim": n_dim,
            "time_dependent": time_dependent,
            "auto_rescale_objective": auto_rescale_objective,
            "exact_hessian": exact_hessian,
            "waypoints": waypoints,
            "waypoint_tolerance_m": waypoint_tolerance_m,
            "waypoint_node_indices": waypoint_node_indices,
            "variable_timestep": waypoints is not None
            if variable_timestep is None
            else variable_timestep,
            "dt_min": dt_min,
            "dt_max": dt_max,
        }
        if dt_max is None:
            _kwargs.pop("dt_max")
        phase_kwargs = {
            "climb_nodes": climb_nodes,
            "descent_nodes": descent_nodes,
        }
        self.init_conditions(**_kwargs, **phase_kwargs)

        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        customized_max_fuel = max_fuel

        X, U = self._build_opti(objective, ts_final_guess=7200, **_kwargs)
        opti = self._opti

        # --- Phase-specific constraints ---

        phase_indices = self._phase_node_indices(**phase_kwargs)
        if phase_indices is not None:
            idx_toc, idx_tod = phase_indices
            cruise_vs_limit = self._cruise_vertical_rate_limit()
            cruise_mach_min = self._cruise_mach_min()

            for k in range(idx_toc, idx_tod):
                # Keep the allocated cruise mesh approximately level while using
                # the model-specific band needed for short-route feasibility.
                opti.subject_to(
                    opti.bounded(-cruise_vs_limit, U[k][1], cruise_vs_limit)  # type: ignore[arg-type]  # CasADi stubs wrong
                )
                # Minimum cruise alt FL150
                opti.subject_to(X[k][2] >= 15000 * ft)
                if cruise_mach_min is not None:
                    opti.subject_to(U[k][0] >= cruise_mach_min)

            for k in range(0, idx_toc):
                opti.subject_to(U[k][1] >= 0)

            for k in range(idx_tod, self.nodes):
                opti.subject_to(U[k][1] <= 0)

        # Force and energy constraints
        for k in range(self.nodes):
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            thrust_max = self._thrust_climb(tas, alt)
            drag = self._constrain_clean_performance(opti, mass, tas, alt, thrust_max)

            # Excess energy > change in potential energy
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[k][1]
            opti.subject_to(excess_energy >= 0)

        # ts and dt consistency
        for k in range(self.nodes - 1):
            opti.subject_to(
                opti.bounded(-1, X[k + 1][4] - X[k][4] - self._interval_dt(k), 1)  # type: ignore[arg-type]
                # CasADi stubs wrong
            )

        # Smooth Mach number change
        mach_change_limit = self._mach_change_limit()
        for k in range(self.nodes - 1):
            mach_delta = U[k + 1][0] - U[k][0]
            opti.subject_to(
                opti.bounded(-mach_change_limit, mach_delta, mach_change_limit)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Limit vertical acceleration independently of interval duration
        for k in range(self.nodes - 1):
            vertical_acceleration = self._control_change_rate(U, k, 1)
            opti.subject_to(
                opti.bounded(
                    -self.MAX_VERTICAL_ACCELERATION,
                    vertical_acceleration,
                    self.MAX_VERTICAL_ACCELERATION,
                )  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Limit turn rate independently of interval duration
        for k in range(self.nodes - 1):
            turn_rate = self._control_change_rate(U, k, 2)
            opti.subject_to(
                opti.bounded(-self.MAX_TURN_RATE, turn_rate, self.MAX_TURN_RATE)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Fuel constraint
        opti.subject_to(opti.bounded(0, X[0][3] - X[-1][3], self.fuel_max))  # type: ignore[arg-type]  # CasADi stubs wrong

        self._constrain_waypoints(
            X,
            waypoints,
            waypoint_tolerance_m=waypoint_tolerance_m,
            waypoint_node_indices=waypoint_node_indices,
        )

        if customized_max_fuel is not None:
            opti.subject_to(X[0][3] - X[-1][3] <= customized_max_fuel)

        # --- Solve ---
        df = self._solve(X, U, **_kwargs)
        df_copy = df.copy()

        if not self._last_solution.stats()["success"]:
            warnings.warn("flight might be infeasible.")

        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.mass_min - self.MASS_CONSTRAINT_TOL_KG:
                warnings.warn(
                    "final mass condition violated (smaller than minimum mass)."
                )
                df = None
            if final_mass > self.mlw + self.MASS_CONSTRAINT_TOL_KG:
                warnings.warn("final mass condition violated (larger than MLW).")
                df = None

        if return_failed:
            df = df_copy

        if result_object:
            return self._make_result(df)
        return df  # type: ignore[return-value]  # df may be None on failed solves; callers handle this
