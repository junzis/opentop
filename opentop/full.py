"""CompleteFlight — full-flight (takeoff to landing) trajectory optimizer."""

from __future__ import annotations

import warnings
from math import pi
from typing import TYPE_CHECKING, Any, Callable

import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from ._continuous_step import ContinuousStepClimb, step_timestep_defaults
from ._path_validation import PhysicalValidationOptions
from ._physical_constraints import energy_row
from ._step_climb import (
    SparseAscent,
    StepClimbOptions,
    resolve_step_climb,
    validate_step_climb_objective,
)
from ._types import LatLon
from .base import Base

if TYPE_CHECKING:
    from ._options import TrajectoryResult
    from .neural import AcropoleFuelFlow, PhysicsCorrectedFuelFlow


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
        fuel_model: AcropoleFuelFlow | PhysicsCorrectedFuelFlow | None = None,
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
            fuel_model=fuel_model,
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
        psi_start, psi_end = self._endpoint_headings(kwargs.get("waypoints"))
        min_mach = 0.3 if self.performance_model == "bada4" else 0.1
        mass_lower = self.mass_min if self.payload is not None else self.oew * 0.5

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, self.mass_init_lb, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_min, self.mass_init_ub, ts_min]

        # Enforce the same landing-mass limits checked after solving.
        if self.mass_min > self.mlw:
            raise ValueError("minimum mass must not exceed maximum landing weight")
        self.x_f_lb = [xp_f, yp_f, h_min, self.mass_min, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, min(self.mass_init_ub, self.mlw), ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_lower, ts_min]
        self.x_ub = [x_max, y_max, h_max, self.mass_init_ub, ts_max]

        # Control init - lower and upper bounds
        self.u_0_lb = [min_mach, 500 * fpm, psi_start]
        self.u_0_ub = [0.3, 2500 * fpm, psi_start]

        # Control final - lower and upper bounds
        self.u_f_lb = [min_mach, -1500 * fpm, psi_end]
        self.u_f_ub = [0.3, -300 * fpm, psi_end]

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
        expand: bool | None = None,
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
            expand=expand,
        )

    def _cruise_vertical_rate_limit(self) -> float:
        """Return the common cruise vertical-rate limit in m/s."""
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
        h_max: float | None = None,
        return_failed: bool = False,
        initial_guess: pd.DataFrame | None = None,
        initialization: str | None = None,
        remove_cruise: bool = False,
        interpolant: Any = None,
        n_dim: int | None = None,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        waypoints: list[LatLon] | None = None,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_min_spacing_m: float = 0.0,
        waypoint_node_indices: list[int] | None = None,
        climb_nodes: int | None = None,
        descent_nodes: int | None = None,
        variable_timestep: bool | None = None,
        dt_min: float | None = None,
        dt_max: float | None = None,
        physical_validation: bool | PhysicalValidationOptions = True,
        result_object: bool = False,
        path_constraint_points: list[tuple[int, float]] | None = None,
        step_climb: bool | StepClimbOptions = False,
    ) -> pd.DataFrame | TrajectoryResult:
        """Compute the optimal complete flight trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            max_fuel: Maximum fuel constraint (kg).
            h_max: Optional altitude ceiling in meters, including between nodes.
                The default FL150 cruise floor becomes 100 ft when a lower
                ceiling applies.
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
            waypoint_min_spacing_m: Skip waypoints closer than this projected
                distance to the last retained point or a flight endpoint, before
                constructing the problem. Default 0 disables thinning. This
                relaxes the supplied route constraints; mandatory fixes must
                not be thinned. The dynamics mesh is unchanged. Retained and
                skipped points are available as active_waypoints and
                skipped_waypoints on the optimizer.
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
            path_constraint_points: Additional (zero-based interval, local time
                fraction in [0, 1]) pairs for force and energy constraints.
                Use points from an independent path audit to constrain missed
                peaks without changing the dynamics mesh. Default None keeps
                boundary and collocation checks. Finite sampling does not
                guarantee feasibility everywhere between the checked points.
            physical_validation: Audit the configured physical constraints at
                161 points per interval and repair at most twice by default.
                False explicitly disables this audit. Failure is returned only
                as a diagnostic frame when return_failed=True.
            step_climb: Optimize optional continuous cruise-climb pulses with
                no minimum climb rate or penalty. Rates, timing, and altitudes
                are free; level dwells and ramp durations remain constrained.
                Initial climb and final descent keep their phase constraints.
                Uses variable timesteps with route/mesh-aware default bounds.
                One solve plus at most two physical repairs is the default;
                there is no global-optimality guarantee. Explicit legacy
                StepClimbOptions retain the older candidate search.
                Requires objective="fuel". Default False.
            initialization: None keeps the legacy ordinary-flight seed and
                selects the coherent pulse seed for continuous step climbs.
                Explicit "auto" opts into phase-based initialization and model
                fuel integration; "legacy" forces the previous seed. Supplied
                initial_guess data takes precedence over both policies.


        Returns:
            pd.DataFrame (or TrajectoryResult if result_object=True).
        """
        step_options = resolve_step_climb(step_climb)
        if initialization is None:
            initialization = (
                "auto"
                if step_options is not None and step_options.method == "continuous"
                else "legacy"
            )
        if initialization not in {"auto", "legacy"}:
            raise ValueError("initialization must be auto or legacy")
        self.initialization_diagnostics = {
            "method": "supplied" if initial_guess is not None else initialization
        }
        validate_step_climb_objective(step_options, objective)
        if (
            step_options is not None
            and step_options.method == "continuous"
            and variable_timestep is False
        ):
            raise ValueError(
                "continuous step_climb requires variable_timestep=True; "
                "use method='legacy' for fixed interval times"
            )
        if step_options is not None:
            variable_timestep = True if variable_timestep is None else variable_timestep
            dt_min, dt_max = step_timestep_defaults(
                self, dt_min, dt_max, method=step_options.method
            )
        waypoints, waypoint_node_indices = self._filter_waypoints(
            waypoints, waypoint_node_indices, waypoint_min_spacing_m
        )
        _kwargs = {
            "initial_guess": initial_guess,
            "physical_validation": physical_validation,
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
        if h_max is not None:
            if not np.isfinite(h_max) or h_max < 100 * ft:
                raise ValueError("h_max must be finite and at least 100 ft")
            _kwargs["h_max"] = min(h_max, self.aircraft["limits"]["ceiling"])
        if dt_max is None:
            _kwargs.pop("dt_max")
        phase_kwargs = {
            "climb_nodes": climb_nodes,
            "descent_nodes": descent_nodes,
        }
        self.init_conditions(**_kwargs, **phase_kwargs)

        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        _kwargs["initialization"] = initialization
        _kwargs["initialization_phase_indices"] = self._phase_node_indices(
            **phase_kwargs
        )
        customized_max_fuel = max_fuel

        X, U = self._build_opti(objective, ts_final_guess=7200, **_kwargs)
        opti = self._opti
        self._last_transcription.physical_contract = "full"
        if step_options is not None and step_options.method == "continuous":
            self._auto_expand = (
                self.performance_model == "openap"
                and (self.wind is None or self._auto_expand)
                and interpolant is None
                and getattr(self, "track_ref", None) is None
                and self.fuel_model is None
            )

        # --- Phase-specific constraints ---

        phase_indices = self._phase_node_indices(**phase_kwargs)
        if phase_indices is not None:
            idx_toc, idx_tod = phase_indices
            cruise_vs_limit = (
                self._cruise_vertical_rate_limit()
                if step_options is None
                else step_options.max_climb_rate_fpm * fpm
            )
            cruise_mach_min = (
                self._cruise_mach_min() if self.x_ub[2] >= 15000 * ft else None
            )

            for k in range(idx_toc, idx_tod + 1):
                # Keep the allocated cruise mesh approximately level while using
                # the model-specific band needed for short-route feasibility.
                opti.subject_to(
                    opti.bounded(-cruise_vs_limit, U[k][1], cruise_vs_limit)  # type: ignore[arg-type]  # CasADi stubs wrong
                )
                # A RAD ceiling below FL150 takes priority over the default floor.
                cruise_floor = 15000 * ft if self.x_ub[2] >= 15000 * ft else 100 * ft
                opti.subject_to(X[k][2] >= cruise_floor)
                if cruise_mach_min is not None:
                    opti.subject_to(U[k][0] >= cruise_mach_min)

            for k in range(0, idx_toc + 1):
                opti.subject_to(U[k][1] >= 0)

            for k in range(idx_tod, self.nodes + 1):
                opti.subject_to(U[k][1] <= 0)

        # Force and energy constraints
        performance_points = list(path_constraint_points or ())
        if step_options is not None:
            performance_points.extend(
                (k, tau) for k in range(self.nodes) for tau in (0.25, 0.5, 0.75)
            )
        for state, control in self._last_transcription.path_points(performance_points):
            mass = state[3]
            v = oc.aero.mach2tas(control[0], state[2], dT=self.dT)
            tas = v / kts
            alt = state[2] / ft
            thrust_max = self._thrust_climb(tas, alt)
            drag = self._constrain_clean_performance(opti, mass, tas, alt, thrust_max)

            # Excess energy > change in potential energy
            energy_row(
                thrust_max,
                drag,
                v,
                mass,
                control[1],
                buffer=getattr(self, "_physical_path_buffer", 0.0),
            ).constrain(opti)

        # ts and dt consistency
        for k in range(self.nodes):
            opti.subject_to(
                opti.bounded(-1, X[k + 1][4] - X[k][4] - self._interval_dt(k), 1)  # type: ignore[arg-type]
                # CasADi stubs wrong
            )

        # Smooth Mach number change
        mach_change_limit = self._mach_change_limit()
        for k in range(self.nodes):
            mach_delta = U[k + 1][0] - U[k][0]
            opti.subject_to(
                opti.bounded(-mach_change_limit, mach_delta, mach_change_limit)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Limit vertical acceleration independently of interval duration
        for k in range(self.nodes):
            vertical_acceleration = self._control_change_rate(U, k, 1)
            opti.subject_to(
                opti.bounded(
                    -self.MAX_VERTICAL_ACCELERATION,
                    vertical_acceleration,
                    self.MAX_VERTICAL_ACCELERATION,
                )  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Limit turn rate independently of interval duration
        for k in range(self.nodes):
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
        self.step_climb_penalty = None
        self.step_climb_stats = []
        self.step_climb_diagnostics = None
        if step_options is None:
            df = self._solve(X, U, **_kwargs)
        else:
            sparse_ascent = (
                ContinuousStepClimb
                if step_options.method == "continuous"
                else SparseAscent
            ).add(self._last_transcription, step_options, idx_toc, idx_tod)
            df = sparse_ascent.solve(self, **_kwargs)
        df_copy = df.copy()

        if not self._last_solution.stats()["success"]:
            warnings.warn("flight might be infeasible.")
            df = None

        if df is not None and df.altitude.max() < 5000 and h_max is None:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            self._reject_solution("Trajectory_Altitude_Check_Failed")
            df = None

        if df is not None:
            final_mass = df.mass.iloc[-1]
            if final_mass < self.mass_min - self.MASS_CONSTRAINT_TOL_KG:
                warnings.warn(
                    "final mass condition violated (smaller than minimum mass)."
                )
                self._reject_solution("Trajectory_Minimum_Mass_Check_Failed")
                df = None
            if final_mass > self.mlw + self.MASS_CONSTRAINT_TOL_KG:
                warnings.warn("final mass condition violated (larger than MLW).")
                self._reject_solution("Trajectory_Landing_Mass_Check_Failed")
                df = None

        if return_failed:
            df = df_copy

        if result_object:
            return self._make_result(df)
        return df  # type: ignore[return-value]  # df may be None on failed solves; callers handle this
