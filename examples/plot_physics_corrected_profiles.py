"""Plot saved original/corrected cruise profiles against OpenAP and BADA 4.

No optimization or external model assets are needed. Output: 150 dpi PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

STYLES = {
    "openap": ("OpenAP", "#666666", "-", "P"),
    "bada4": ("BADA 4", "#009E73", "-.", "s"),
    "acropole": ("Acropole: original", "#D55E00", "--", "o"),
    "wave_increment": ("Acropole: OpenAP wave correction", "#CC79A7", ":", "^"),
    "speed_increment": ("Acropole: OpenAP speed correction", "#0072B2", "-", "D"),
}


def load_profiles(directory, report):
    if report["configuration"]["correction_model"] != "openap":
        raise ValueError("Expected corrections constructed from OpenAP.")
    reference = report["configuration"]["reference_mach"]
    profiles = {}
    for force in ("openap", "bada4"):
        for model in STYLES:
            records = [
                r
                for r in report["solves"]
                if r["force_model"] == force
                and r["model"] == model
                and r["run"] == "main"
            ]
            if len(records) != 1 or not records[0]["success"]:
                raise ValueError(f"Expected one successful {force}/{model} main solve.")
            record = records[0]
            frame = pd.read_csv(directory / f"{force}_{model}_main_{reference:.2f}.csv")
            required = frame[["ts", "mass", "altitude", "mach", "fuelflow"]]
            if (
                not np.isfinite(required).all().all()
                or not (np.diff(frame.ts) > 0).all()
            ):
                raise ValueError(
                    f"Non-finite values or non-increasing time: {force}/{model}"
                )
            # fuel_cost describes intervals; the final row has no next interval.
            if not np.isfinite(frame.fuel_cost.iloc[:-1]).all():
                raise ValueError(f"Missing interval fuel cost: {force}/{model}")
            np.testing.assert_allclose(
                frame.mass.iloc[0] - frame.mass.iloc[-1],
                record["fuel_kg"],
                rtol=0,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                frame.fuel_cost.sum(), record["fuel_kg"], rtol=0, atol=1e-5
            )
            np.testing.assert_allclose(
                frame.ts.iloc[-1], record["duration_s"], rtol=0, atol=1e-4
            )
            profiles[force, model] = frame
    return profiles


def plot_profiles(profiles, report, force, output):
    force_label = "OpenAP" if force == "openap" else "BADA 4"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    fig.subplots_adjust(
        left=0.10, right=0.96, top=0.70, bottom=0.17, hspace=0.42, wspace=0.25
    )
    fig.text(
        0.10,
        0.96,
        f"Optimized cruise profiles: {force_label} force constraints",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.10,
        0.917,
        "A320 | EHAM to EDDF | Initial mass 66,300 kg | Still air",
        color="#536572",
        fontsize=11,
    )
    for index, (model, (label, color, style, marker)) in enumerate(STYLES.items()):
        df = profiles[force, model]
        elapsed = df.ts / 60
        burnt = df.mass.iloc[0] - df.mass
        # Different marker positions make nearly coincident fuel curves visible.
        positions = sorted(set([index % 3, len(df) // 2 + index % 3, len(df) - 1]))
        for ax, values in zip(axes.flat, (df.altitude, df.mach, df.fuelflow, burnt)):
            ax.plot(
                elapsed,
                values,
                label=label,
                color=color,
                ls=style,
                lw=1.8,
                marker=marker,
                ms=4.5,
                markevery=positions,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.09, 0.875),
        frameon=False,
        ncol=2,
        handlelength=3.0,
        columnspacing=2.5,
    )
    titles = (
        "A   Optimized altitude",
        "B   Optimized Mach",
        "C   Predicted fuel flow",
        "D   Cumulative predicted fuel burn",
    )
    ylabels = ("Altitude (ft)", "Mach number", "Fuel flow (kg/s)", "Fuel burn (kg)")
    # Common scales across the two figures permit direct visual comparison.
    frames = list(profiles.values())
    max_time = max(df.ts.iloc[-1] for df in frames) / 60
    max_fuel = max(df.mass.iloc[0] - df.mass.iloc[-1] for df in frames)
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set(title=title, ylabel=ylabel, xlim=(0, max_time * 1.035))
        ax.title.set_weight("bold")
        ax.grid(axis="y", color="#DDE3E8", lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylim(31000, 35000)
    axes[0, 0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0, 1].set_ylim(0.725, 0.835)
    axes[0, 1].axhline(0.82, color="#A0A0A0", ls=":", lw=0.8, zorder=0)
    axes[0, 1].text(0.3, 0.824, "M 0.82 limit", color="#687781", fontsize=8)
    axes[1, 0].set_ylim(
        min(df.fuelflow.min() for df in frames) - 0.015,
        max(df.fuelflow.max() for df in frames) + 0.015,
    )
    axes[1, 1].set_ylim(0, max_fuel * 1.06)
    for ax in axes[1]:
        ax.set_xlabel("Elapsed cruise time (min)")
    config = report["configuration"]
    reference, width = config["reference_mach"], config["transition_width"]
    fig.text(
        0.10,
        0.065,
        "Cruise only: altitude and Mach were constrained to be constant; "
        "endpoints show different flight durations.\n"
        f"Both corrections use OpenAP; speed transition M {reference:.2f} "
        f"to {reference + width:.2f}. "
        "Fuel predictions use each curve's own model, not measured fuel.",
        fontsize=9,
        color="#536572",
        linespacing=1.6,
    )
    target = output / f"optimized_profiles_{force}.png"
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.results_dir / "report.json").read_text())
    profiles = load_profiles(args.results_dir, report)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#89939A",
            "axes.linewidth": 0.7,
        }
    )
    for force in ("openap", "bada4"):
        plot_profiles(profiles, report, force, output)


if __name__ == "__main__":
    main()
