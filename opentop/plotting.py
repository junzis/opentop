"""Shared scientific-figure styling for OpenTOP examples and applications."""

from __future__ import annotations

from collections.abc import Iterable
from string import ascii_uppercase
from typing import Any

import matplotlib as mpl
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cycler import cycler
from matplotlib.patches import FancyBboxPatch

OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

LAND_COLOR = "#F4F3EF"
OCEAN_COLOR = "#EAF2F8"
COAST_COLOR = "#555555"
GRID_COLOR = "#777777"
NEUTRAL_COLOR = "#777777"


def apply_publication_style(*, dpi: int = 120) -> None:
    """Apply a compact, colorblind-safe Matplotlib style.

    The defaults target readable notebook and double-column scientific figures.
    They intentionally avoid a named system font so headless execution remains
    deterministic across Linux, macOS, and Windows.
    """

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "legend.fontsize": 8,
            "legend.framealpha": 0.82,
            "legend.edgecolor": "#B0B0B0",
            "legend.fancybox": True,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "figure.dpi": dpi,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "image.cmap": "viridis",
        }
    )


def style_axes(axis: Any, *, grid: bool = True, legend: bool = True) -> Any:
    """Apply consistent finishing details to a Cartesian axis."""

    for name in ("top", "right"):
        spine = axis.spines.get(name)
        if spine is not None:
            spine.set_visible(False)
    axis.grid(grid)
    if legend:
        current_legend = axis.get_legend()
        if current_legend is not None:
            current_legend.set_frame_on(True)
            current_legend.get_frame().set_alpha(0.82)
    return axis


def _flatten_axes(axes: Any) -> list[Any]:
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    if isinstance(axes, Iterable) and not isinstance(axes, (str, bytes)):
        flattened: list[Any] = []
        for item in axes:
            flattened.extend(_flatten_axes(item))
        return flattened
    return [axes]


def add_panel_labels(
    axes: Any,
    *,
    labels: str = ascii_uppercase,
    x: float = -0.12,
    y: float = 1.05,
) -> None:
    """Add bold panel labels to an axis or nested collection of axes."""

    visible_axes = [axis for axis in _flatten_axes(axes) if axis.get_visible()]
    for label, axis in zip(labels, visible_axes):
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )


def style_map(
    axis: Any,
    *,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    data_crs: Any | None = None,
    title: str | None = None,
    resolution: str = "50m",
    labels: bool = True,
) -> Any:
    """Apply consistent Cartopy land, ocean, border, and grid styling."""

    data_crs = ccrs.PlateCarree() if data_crs is None else data_crs
    if extent is not None:
        axis.set_extent(extent, crs=data_crs)
    axis.add_feature(
        cfeature.LAND.with_scale(resolution), facecolor=LAND_COLOR, zorder=0
    )
    axis.add_feature(
        cfeature.OCEAN.with_scale(resolution), facecolor=OCEAN_COLOR, zorder=0
    )
    axis.coastlines(resolution=resolution, linewidth=0.55, color=COAST_COLOR)
    axis.add_feature(
        cfeature.BORDERS.with_scale(resolution),
        linewidth=0.45,
        edgecolor=GRID_COLOR,
    )
    gridlines = axis.gridlines(
        crs=data_crs,
        draw_labels=labels,
        linewidth=0.35,
        color=GRID_COLOR,
        alpha=0.45,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    if labels:
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 7}
        gridlines.ylabel_style = {"size": 7}
    if title is not None:
        axis.set_title(title)
    legend = axis.get_legend()
    if legend is not None:
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(0.82)
    return axis


def add_wind_vector_key(
    axis: Any,
    vectors: Any,
    *,
    reference_mps: float = 20.0,
    x: float = 0.76,
    y: float = 0.06,
) -> Any:
    """Add a wind-vector scale with a semi-transparent background."""

    background = FancyBboxPatch(
        (x - 0.14, y - 0.035),
        0.30,
        0.08,
        boxstyle="round,pad=0.012",
        transform=axis.transAxes,
        facecolor="white",
        edgecolor="#B0B0B0",
        linewidth=0.6,
        alpha=0.80,
        zorder=8,
        clip_on=False,
    )
    axis.add_patch(background)
    key = axis.quiverkey(
        vectors,
        x,
        y,
        reference_mps,
        f"{reference_mps:g} m/s",
        coordinates="axes",
        labelpos="E",
    )
    key.set_zorder(9)
    return key


def plot_wind_vectors(
    axis: Any,
    wind: Any,
    *,
    data_crs: Any | None = None,
    scale: float = 300.0,
    width: float = 0.0035,
    reference_mps: float = 20.0,
    key_x: float = 0.76,
    key_y: float = 0.06,
) -> Any:
    """Plot east/north wind components and add a readable scale key."""

    data_crs = ccrs.PlateCarree() if data_crs is None else data_crs
    vectors = axis.quiver(
        wind.longitude,
        wind.latitude,
        wind.u,
        wind.v,
        color=OKABE_ITO[0],
        scale=scale,
        width=width,
        transform=data_crs,
        zorder=4,
    )
    add_wind_vector_key(
        axis,
        vectors,
        reference_mps=reference_mps,
        x=key_x,
        y=key_y,
    )
    return vectors


def route_style(index: int) -> dict[str, Any]:
    """Return redundant color, line-style, and marker encodings."""

    return {
        "color": OKABE_ITO[index % len(OKABE_ITO)],
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
    }
