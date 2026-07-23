"""Regenerate the compact FastMeteo/ERA5 route-weather samples.

The checked-in wind CSV and contrail-cost interpolant keep the examples
deterministic and runnable without a network connection. Regeneration requires
FastMeteo access to its ERA5 source and the OpenTOP replay dependencies.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).with_name("route_wind_era5_20210501.csv")
CONTRAIL_OUTPUT = Path(__file__).with_name("route_contrail_era5_20210501.casadi")
DEFAULT_STORE = "/tmp/opentop-example-era5-zarr"


def build_fastmeteo_weather_sample(
    *,
    local_store: str = DEFAULT_STORE,
    include_vertical_boundaries: bool = False,
) -> pd.DataFrame:
    """Interpolate wind, temperature, and humidity on the fixed example grid."""

    from fastmeteo.source import ArcoEra5

    altitude_levels = [1_000, 9_000, 17_000, 25_000, 35_000, 41_000]
    if include_vertical_boundaries:
        altitude_levels = [0, *altitude_levels, 45_000]
    latitudes, longitudes, altitudes, timestamps = np.meshgrid(
        np.linspace(40.0, 54.0, 8),
        np.linspace(2.0, 16.0, 8),
        np.asarray(altitude_levels),
        pd.date_range("2021-05-01 08:00:00", periods=7, freq="1h"),
    )
    grid = pd.DataFrame(
        {
            "latitude": latitudes.ravel(),
            "longitude": longitudes.ravel(),
            "altitude": altitudes.ravel(),
            "timestamp": timestamps.ravel(),
        }
    )

    return ArcoEra5(local_store=local_store).interpolate(grid)


def wind_from_weather(meteo: pd.DataFrame) -> pd.DataFrame:
    """Convert the FastMeteo output to the columns expected by OpenTOP."""

    return (
        meteo.rename(
            columns={
                "u_component_of_wind": "u",
                "v_component_of_wind": "v",
            }
        )
        .assign(
            ts=lambda frame: (
                frame["timestamp"] - frame["timestamp"].min()
            ).dt.total_seconds(),
            h=lambda frame: frame["altitude"] * 0.3048,
        )[["longitude", "latitude", "h", "ts", "u", "v"]]
        .sort_values(["ts", "h", "latitude", "longitude"])
    )


def build_fastmeteo_wind_sample(*, local_store: str = DEFAULT_STORE) -> pd.DataFrame:
    """Build the fixed wind sample from ERA5 through FastMeteo."""

    return wind_from_weather(build_fastmeteo_weather_sample(local_store=local_store))


def build_fastmeteo_contrail_sample(
    *,
    path: Path = CONTRAIL_OUTPUT,
    local_store: str = DEFAULT_STORE,
    sigma: int = 2,
):
    """Build and cache a 4-D persistent-contrail cost interpolant."""

    from opentop import replay, tools

    meteo = build_fastmeteo_weather_sample(
        local_store=local_store,
        include_vertical_boundaries=True,
    )
    interpolant = replay.build_contrail_interpolant(meteo, sigma=sigma)
    tools.save_interpolant(interpolant, path)
    print(f"Wrote contrail-cost interpolant to {path}")
    return interpolant


def load_fastmeteo_wind_sample(
    *,
    refresh: bool = False,
    path: Path = OUTPUT,
    local_store: str = DEFAULT_STORE,
) -> pd.DataFrame:
    """Load the cached sample, optionally refreshing it through FastMeteo."""

    if not refresh:
        return pd.read_csv(path)

    wind = build_fastmeteo_wind_sample(local_store=local_store)
    wind.to_csv(path, index=False, float_format="%.6f")
    print(f"Wrote {len(wind)} wind samples to {path}")
    return wind


def load_fastmeteo_contrail_sample(
    *,
    refresh: bool = False,
    path: Path = CONTRAIL_OUTPUT,
    local_store: str = DEFAULT_STORE,
    sigma: int = 2,
):
    """Load the cached contrail field, optionally refreshing it from ERA5."""

    from opentop import tools

    if refresh:
        return build_fastmeteo_contrail_sample(
            path=path,
            local_store=local_store,
            sigma=sigma,
        )
    return tools.load_interpolant(path)


def main() -> None:
    load_fastmeteo_wind_sample(refresh=True)
    load_fastmeteo_contrail_sample(refresh=True)


if __name__ == "__main__":
    main()
