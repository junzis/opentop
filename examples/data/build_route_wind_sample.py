"""Regenerate the compact FastMeteo/ERA5 wind sample used by route notebooks.

The checked-in CSV keeps the examples deterministic and runnable without a
network connection. Regeneration requires FastMeteo access to its ERA5 source.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).with_name("route_wind_era5_20210501.csv")
DEFAULT_STORE = "/tmp/opentop-example-era5-zarr"


def build_fastmeteo_wind_sample(*, local_store: str = DEFAULT_STORE) -> pd.DataFrame:
    """Interpolate the fixed example grid from ERA5 through FastMeteo."""

    from fastmeteo.source import ArcoEra5

    latitudes, longitudes, altitudes, timestamps = np.meshgrid(
        np.linspace(40.0, 54.0, 8),
        np.linspace(2.0, 16.0, 8),
        np.array([1_000, 9_000, 17_000, 25_000, 35_000, 41_000]),
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

    meteo = ArcoEra5(local_store=local_store).interpolate(grid)
    wind = (
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
    return wind


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


def main() -> None:
    load_fastmeteo_wind_sample(refresh=True)


if __name__ == "__main__":
    main()
