from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, Any, Callable

import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np
import pandas as pd

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
    ) -> None:
        """Initialize direct collocation bounds and guesses.

        Args:
            df_cruise: Cruise trajectory DataFrame.
            alt_start: Start altitude in feet. If provided, used instead of df_cruise.
        """

        h_min = 100 * ft
        od_psi = self._compute_bearing_psi()

        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)

        x_min, x_max, y_min, y_max = self._compute_bbox(waypoints=waypoints)
        ts_min = 0
        ts_max = 6 * 3600

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
        xp_guess = np.linspace(xp_0, xp_f, self.nodes + 1)
        yp_guess = np.linspace(yp_0, yp_f, self.nodes + 1)
        h_guess = np.linspace(h_start, h_min, self.nodes + 1)
        m_guess = mass_tod * np.ones(self.nodes + 1)
        ts_guess = np.linspace(0, 6 * 3600, self.nodes + 1)
        self.x_guess = np.vstack([xp_guess, yp_guess, h_guess, m_guess, ts_guess]).T

        # Control init - lower and upper bounds
        self.u_0_lb = [cruise_mach - 0.1, -2500 * fpm, od_psi - pi / 4]
        self.u_0_ub = [cruise_mach + 0.1, 0 * fpm, od_psi + pi / 4]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.1, -2500 * fpm, od_psi - pi / 4]
        self.u_f_ub = [0.3, 0 * fpm, od_psi + pi / 4]

        # Control - Lower and upper bound
        self.u_lb = [0.1, -2500 * fpm, od_psi - pi / 2]
        self.u_ub = [cruise_mach, 0 * fpm, od_psi + pi / 2]

        # Control - guesses
        self.u_guess = [0.7, 0 * fpm, od_psi]

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
        result_object: bool = False,
    ) -> pd.DataFrame | TrajectoryResult:
        """Compute the optimal descent trajectory.

        Args:
            objective: Optimization objective. Default "fuel".
            df_cruise: Cruise trajectory for initial altitude/mach. If None,
                computed automatically.
            alt_start: Start of descent altitude in feet.
            remove_cruise: Remove level-off points. Default True.
            initial_guess: Unused by Descent (accepted for API symmetry).
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
        }
        if dt_max is None:
            _kwargs.pop("dt_max")

        self.init_conditions(df_cruise, alt_start=alt_start, waypoints=waypoints)

        X, U = self._build_opti(objective, ts_final_guess=3600, **_kwargs)
        opti = self._opti

        # --- Phase-specific constraints ---

        # Constrain time and dt
        for k in range(1, self.nodes):
            time_delta = X[k][4] - X[k - 1][4] - self._interval_dt(k - 1)
            opti.subject_to(opti.bounded(-1, time_delta, 1))  # type: ignore[arg-type]  # CasADi stubs wrong

        # Smooth Mach number changes
        for k in range(1, self.nodes):
            opti.subject_to(opti.bounded(-0.1, U[k][0] - U[k - 1][0], 0.1))  # type: ignore[arg-type]  # CasADi stubs wrong

        # Smooth vertical rate changes
        for k in range(1, self.nodes):
            opti.subject_to(
                opti.bounded(-1000 * fpm, U[k][1] - U[k - 1][1], 1000 * fpm)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Smooth heading changes
        for k in range(1, self.nodes):
            opti.subject_to(
                opti.bounded(-5 * pi / 180, U[k][2] - U[k - 1][2], 5 * pi / 180)  # type: ignore[arg-type]  # CasADi stubs wrong
            )

        # Force and energy constraints
        for k in range(self.nodes):
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2], dT=self.dT)
            tas = v / kts
            alt = X[k][2] / ft
            thrust_max = self.thrust.cruise(tas, alt, dT=self.dT)
            drag = self._constrain_clean_performance(opti, mass, tas, alt, thrust_max)

            # Excess energy > change in potential energy
            excess_energy = (thrust_max - drag) * v - mass * oc.aero.g0 * U[k][1]
            opti.subject_to(excess_energy >= 0)

        self._constrain_waypoints(
            X,
            waypoints,
            waypoint_tolerance_m=waypoint_tolerance_m,
            waypoint_node_indices=waypoint_node_indices,
        )

        # --- Solve ---
        df = self._solve(X, U, **_kwargs)

        if remove_cruise:
            df = df.query("vertical_rate < -100")

        if result_object:
            return self._make_result(df)
        return df
