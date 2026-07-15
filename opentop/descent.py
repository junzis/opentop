from __future__ import annotations

from itertools import pairwise
from math import pi
from typing import TYPE_CHECKING, Any, Callable

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

from ._transcription import AircraftTranscription
from ._types import LatLon
from .base import Base
from .cruise import Cruise

if TYPE_CHECKING:
    from ._options import TrajectoryResult


class Descent(Base):
    """Descent phase trajectory optimizer."""

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
        )
        self.cruise = Cruise(
            actype,
            origin,
            destination,
            m0=m0,
            engine=engine,
            use_synonym=use_synonym,
            dT=dT,
            performance_model=performance_model,
            bada_path=bada_path,
        )

    def init_conditions(
        self,
        df_cruise: pd.DataFrame,
        alt_start: float | None = None,
        waypoints: list[LatLon] | None = None,
        waypoint_node_indices: list[int] | None = None,
        route_margin_m: float = 10_000.0,
        max_duration_s: float | None = None,
    ) -> None:
        """Initialize direct collocation bounds and guesses.

        Args:
            df_cruise: Cruise trajectory DataFrame.
            alt_start: Start altitude in feet. If provided, used instead of df_cruise.
        """

        h_min = 100 * ft
        od_psi = self._compute_bearing_psi()
        normalized_waypoints = self._normalize_waypoints(waypoints)
        route_points = [
            (self.lat1, self.lon1),
            *normalized_waypoints,
            (self.lat2, self.lon2),
        ]
        route_headings = []
        for (lat_a, lon_a), (lat_b, lon_b) in pairwise(route_points):
            heading = float(oc.geo.bearing(lat_a, lon_a, lat_b, lon_b)) * pi / 180
            # Keep periodic heading representations close to the direct-route
            # bearing so the combined bounds do not imply a full turn.
            heading += 2 * pi * round((od_psi - heading) / (2 * pi))
            route_headings.append(heading)
        initial_psi = route_headings[0]
        final_psi = route_headings[-1]

        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)

        x_min, x_max, y_min, y_max = self._compute_bbox(
            margin_m=route_margin_m, waypoints=waypoints
        )
        ts_min = 0
        if max_duration_s is not None and (
            not np.isfinite(max_duration_s) or max_duration_s <= 0
        ):
            raise ValueError("max_duration_s must be finite and positive")
        ts_max = 6 * 3600 if max_duration_s is None else max_duration_s

        mass_oew = self.oew
        mass_tod = df_cruise.mass.iloc[0]
        cruise_mach = df_cruise.mach.max()

        if alt_start is not None:
            h_start = alt_start * ft
            if h_start > df_cruise.h.iloc[0]:
                print(
                    "The given alt_start is beyond performance limit, "
                    f"we will use {df_cruise.h.iloc[0] / ft}"
                )
                h_start = df_cruise.h.iloc[0]
        else:
            h_start = df_cruise.h.iloc[0]

        # Initial conditions - Lower and upper bounds
        self.x_0_lb = [xp_0 - 1000, yp_0 - 1000, h_start - 100, mass_tod, ts_min]
        self.x_0_ub = [xp_0 + 1000, yp_0 + 1000, h_start + 100, mass_tod, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, mass_oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_min, mass_tod, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, mass_oew, ts_min]
        self.x_ub = [x_max, y_max, h_start + 100, mass_tod, ts_max]

        # States - guesses
        # dist = h_tod / np.tan(np.radians(3))  # 3 deg
        # xp_guess = xp_f - np.linspace(dist * np.sin(od_psi), 0, self.nodes + 1)
        # yp_guess = yp_f - np.linspace(dist * np.cos(od_psi), 0, self.nodes + 1)
        route_indices = self._waypoint_node_indices(
            normalized_waypoints, waypoint_node_indices
        )
        route_xy = [(xp_0, yp_0)]
        route_xy.extend(self.proj(lon, lat) for lat, lon in normalized_waypoints)
        route_xy.append((xp_f, yp_f))
        anchor_nodes = [0, *route_indices, self.nodes]
        self._route_headings = route_headings
        self._route_anchor_nodes = anchor_nodes
        self._route_xy = route_xy
        xp_guess = np.empty(self.nodes + 1)
        yp_guess = np.empty(self.nodes + 1)
        for start, end, point_a, point_b in zip(
            anchor_nodes[:-1], anchor_nodes[1:], route_xy[:-1], route_xy[1:]
        ):
            xp_guess[start : end + 1] = np.linspace(
                point_a[0], point_b[0], end - start + 1
            )
            yp_guess[start : end + 1] = np.linspace(
                point_a[1], point_b[1], end - start + 1
            )
        h_guess = np.linspace(h_start, h_min, self.nodes + 1)
        m_guess = mass_tod * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, min(3600, ts_max), self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [cruise_mach - 0.1, -2500 * fpm, initial_psi - pi / 4]
        self.u_0_ub = [cruise_mach + 0.1, 0 * fpm, initial_psi + pi / 4]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -2500 * fpm, final_psi - pi / 4]
        self.u_f_ub = [0.3, 0 * fpm, final_psi + pi / 4]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, min(route_headings) - pi / 2]
        self.u_ub = [cruise_mach, 0 * fpm, max(route_headings) + pi / 2]

        # Control - guesses
        self.u_guess = [0.7, 0 * fpm, initial_psi]

    def _add_formulation(
        self,
        opti: ca.Opti,
        objective: str | Callable = "fuel",
        *,
        df_cruise: pd.DataFrame | None = None,
        alt_start: float | None = None,
        initial_guess: pd.DataFrame | None = None,
        interpolant: Any = None,
        n_dim: int | None = None,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        waypoints: list[LatLon] | None = None,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_node_indices: list[int] | None = None,
        variable_timestep: bool | None = None,
        dt_min: float | None = None,
        dt_max: float | None = None,
        route_margin_m: float = 10_000.0,
        route_heading_tolerance_deg: float | None = None,
        inbound_route_side: str | None = None,
        max_duration_s: float | None = None,
        name_prefix: str = "flight",
        minimize: bool = False,
    ) -> AircraftTranscription:
        """Add this descent phase to an existing CasADi Opti stack."""
        kwargs = {
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
            "route_margin_m": route_margin_m,
        }
        if dt_max is None:
            kwargs.pop("dt_max")

        conditions = df_cruise
        if conditions is None:
            if initial_guess is not None:
                conditions = initial_guess
            else:
                conditions = self.cruise.trajectory(objective)  # type: ignore[assignment]
        assert isinstance(conditions, pd.DataFrame)
        self.init_conditions(
            conditions,
            alt_start=alt_start,
            waypoints=waypoints,
            waypoint_node_indices=waypoint_node_indices,
            route_margin_m=route_margin_m,
            max_duration_s=max_duration_s,
        )
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        if initial_guess is not None:
            ts_final_guess = float(initial_guess.ts.iloc[-1] - initial_guess.ts.iloc[0])
        else:
            ts_final_guess = min(3600.0, max_duration_s or 3600.0)
        transcription = self._add_transcription(
            opti,
            objective,
            ts_final_guess=ts_final_guess,
            minimize=minimize,
            name_prefix=name_prefix,
            **kwargs,
        )
        X, U = transcription.X, transcription.U

        # Constrain time and dt.
        for k in range(1, self.nodes):
            time_delta = X[k][4] - X[k - 1][4] - self._interval_dt(k - 1)
            opti.subject_to(opti.bounded(-1, time_delta, 1))  # type: ignore[arg-type]

        # Smooth Mach number changes.
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-0.1, U[k][0] - U[k - 1][0], 0.1))  # type: ignore[arg-type]

        # Limit vertical acceleration and turn rate independently of dt.
        for k in range(1, self.nodes):
            vertical_acceleration = self._control_change_rate(U, k - 1, 1)
            opti.subject_to(
                opti.bounded(
                    -self.MAX_VERTICAL_ACCELERATION,
                    vertical_acceleration,
                    self.MAX_VERTICAL_ACCELERATION,
                )  # type: ignore[arg-type]
            )
            turn_rate = self._control_change_rate(U, k - 1, 2)
            opti.subject_to(
                opti.bounded(-self.MAX_TURN_RATE, turn_rate, self.MAX_TURN_RATE)  # type: ignore[arg-type]
            )

        # Force and energy constraints.
        for k in range(self.nodes):
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self._constrain_clean_performance(opti, mass, tas, alt, thrust_max)
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[k][1]
            opti.subject_to(excess_energy >= 0)

        self._constrain_waypoints(
            X,
            waypoints,
            waypoint_tolerance_m=waypoint_tolerance_m,
            waypoint_node_indices=waypoint_node_indices,
        )
        self._constrain_route_headings(
            opti,
            U,
            route_heading_tolerance_deg=route_heading_tolerance_deg,
        )
        self._constrain_inbound_route_side(
            opti,
            X,
            inbound_route_side=inbound_route_side,
        )
        return transcription

    def _constrain_inbound_route_side(
        self,
        opti: ca.Opti,
        X: list[Any],
        *,
        inbound_route_side: str | None,
    ) -> None:
        """Keep the inbound path on one side of its nominal route leg."""
        if inbound_route_side is None:
            return
        if not isinstance(inbound_route_side, str):
            raise ValueError("inbound_route_side must be 'left', 'right', or None")
        side = inbound_route_side.lower()
        if side not in {"left", "right"}:
            raise ValueError("inbound_route_side must be 'left', 'right', or None")

        (x_start, y_start), (x_end, y_end) = self._route_xy[:2]
        dx = x_end - x_start
        dy = y_end - y_start
        side_sign = 1.0 if side == "left" else -1.0
        first_anchor = self._route_anchor_nodes[1]
        for node in range(1, first_anchor):
            cross_track = dx * (X[node][1] - y_start) - dy * (X[node][0] - x_start)
            opti.subject_to(side_sign * cross_track >= 0)

    def _constrain_route_headings(
        self,
        opti: ca.Opti,
        U: list[Any],
        *,
        route_heading_tolerance_deg: float | None,
    ) -> None:
        """Keep controls aligned with straight route legs."""
        if route_heading_tolerance_deg is None:
            return
        if (
            not np.isfinite(route_heading_tolerance_deg)
            or route_heading_tolerance_deg <= 0
            or route_heading_tolerance_deg > 180
        ):
            raise ValueError(
                "route_heading_tolerance_deg must be finite and in (0, 180]"
            )
        tolerance = float(np.deg2rad(route_heading_tolerance_deg))
        for heading, start, end in zip(
            self._route_headings,
            self._route_anchor_nodes[:-1],
            self._route_anchor_nodes[1:],
        ):
            for node in range(start, end):
                opti.subject_to(
                    opti.bounded(
                        heading - tolerance,
                        U[node][2],
                        heading + tolerance,
                    )  # type: ignore[arg-type]
                )

    def trajectory(
        self,
        objective: str | Callable = "fuel",
        df_cruise: pd.DataFrame | None = None,
        *,
        alt_start: float | None = None,
        remove_cruise: bool = True,
        initial_guess: pd.DataFrame | None = None,
        interpolant: Any = None,
        n_dim: int | None = None,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        waypoints: list[LatLon] | None = None,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_node_indices: list[int] | None = None,
        variable_timestep: bool | None = None,
        dt_min: float | None = None,
        dt_max: float | None = None,
        route_margin_m: float = 10_000.0,
        route_heading_tolerance_deg: float | None = None,
        inbound_route_side: str | None = None,
        max_duration_s: float | None = None,
        result_object: bool = False,
    ) -> pd.DataFrame | TrajectoryResult:
        """Compute the optimal descent trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            df_cruise: Cruise trajectory for initial altitude/mach. If None,
                computed automatically.
            alt_start: Start of descent altitude in feet.
            remove_cruise: Remove level-off points. Default True.
            initial_guess: Existing descent trajectory used as the initial guess.
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
            variable_timestep: Optimize interval durations. Defaults to True
                when waypoints are supplied, otherwise False.
            dt_min: Minimum interval duration in seconds for variable timesteps.
                Defaults to an automatic fraction of the expected interval duration.
            dt_max: Maximum interval duration in seconds for variable timesteps.
            route_margin_m: Lateral projected-coordinate bound around the route.
            route_heading_tolerance_deg: Optional heading tolerance around each
                straight route segment, in degrees.
            inbound_route_side: Optionally keep the inbound path on the
                ``"left"`` or ``"right"`` side of its nominal route leg.
            max_duration_s: Optional upper bound on total descent duration.
            result_object: If True, return a TrajectoryResult.

        Returns:
            pd.DataFrame (or TrajectoryResult if result_object=True).
        """
        if df_cruise is None:
            if self.debug:
                print("Finding the preliminary optimal cruise parameters...")
            df_cruise = self.cruise.trajectory(objective)  # type: ignore[assignment]  # result_object=False always returns DataFrame

        if self.debug:
            print("Calculating optimal descent trajectory...")

        assert isinstance(df_cruise, pd.DataFrame)

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
            "route_margin_m": route_margin_m,
            "route_heading_tolerance_deg": route_heading_tolerance_deg,
            "inbound_route_side": inbound_route_side,
            "max_duration_s": max_duration_s,
        }
        if dt_max is None:
            _kwargs.pop("dt_max")

        opti = ca.Opti()
        transcription = self._add_formulation(
            opti,
            objective,
            df_cruise=df_cruise,
            alt_start=alt_start,
            initial_guess=initial_guess,
            interpolant=interpolant,
            n_dim=n_dim,
            time_dependent=time_dependent,
            auto_rescale_objective=auto_rescale_objective,
            exact_hessian=exact_hessian,
            waypoints=waypoints,
            waypoint_tolerance_m=waypoint_tolerance_m,
            waypoint_node_indices=waypoint_node_indices,
            variable_timestep=variable_timestep,
            dt_min=dt_min,
            dt_max=dt_max,
            route_margin_m=route_margin_m,
            route_heading_tolerance_deg=route_heading_tolerance_deg,
            inbound_route_side=inbound_route_side,
            max_duration_s=max_duration_s,
            minimize=True,
        )

        df = self._solve(transcription.X, transcription.U, **_kwargs)

        if remove_cruise:
            df = df.query("vertical_rate < -100")

        if result_object:
            return self._make_result(df)
        return df
