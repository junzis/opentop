"""Plot outputs from neural_fuel.py without rerunning the optimizer.

uv run python examples/plot_neural_fuel.py \
    --results-dir /tmp/opentop-neural-fuel-exact \
    --output-dir examples/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results_dir / "report.json").read_text())
    baseline = report["experiment"].get("force_model", "openap")
    baseline_label = {"openap": "OpenAP", "bada4": "BADA 4"}[baseline]
    if not all(item["success"] for item in report["solves"].values()):
        raise ValueError("Only successful optimization results should be plotted.")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#88929C",
            "axes.linewidth": 0.7,
            "xtick.color": "#44515C",
            "ytick.color": "#44515C",
            "text.color": "#172B3A",
            "axes.labelcolor": "#172B3A",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), sharex=True)
    fig.subplots_adjust(
        left=0.11, right=0.95, bottom=0.20, top=0.76, hspace=0.43, wspace=0.32
    )
    fig.text(
        0.11,
        0.95,
        f"Cruise optimization with {baseline_label} and Acropole",
        fontsize=16,
        weight="bold",
    )
    exp = report["experiment"]
    fig.text(
        0.11,
        0.909,
        f"{exp['aircraft']}  |  {exp['origin']} to {exp['destination']}"
        "  |  Still air; constant altitude and Mach",
        fontsize=10.5,
        color="#536572",
    )
    colors = {baseline: "#0072B2", "acropole": "#D55E00"}
    rates = []
    max_time = 0
    for name, label, style, marker in (
        (baseline, baseline_label, "-", "o"),
        ("acropole", "Acropole neural network", "--", "s"),
    ):
        df = pd.read_csv(args.results_dir / f"{name}.csv")
        time = df.ts / 60
        max_time = max(max_time, time.max())
        rates.extend(df.fuelflow)
        burnt = df.mass.iloc[0] - df.mass
        for ax, values in zip(axes.flat, (df.altitude, df.mach, df.fuelflow, burnt)):
            ax.plot(
                time,
                values,
                color=colors[name],
                ls=style,
                lw=1.9,
                marker=marker,
                ms=3.1,
                markevery=3,
                label=label,
            )
        total = report["solves"][name]["fuel_kg"]
        axes[1, 1].annotate(
            f"{total:,.0f} kg",
            (time.iloc[-1], burnt.iloc[-1]),
            xytext=(
                -6,
                8
                if total == max(s["fuel_kg"] for s in report["solves"].values())
                else -16,
            ),
            textcoords="offset points",
            ha="right",
            color=colors[name],
            fontsize=9,
            weight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.873),
        frameon=False,
        ncol=2,
        handlelength=3,
        columnspacing=2.5,
    )
    titles = [
        "A   Selected altitude",
        "B   Selected Mach",
        "C   Predicted fuel flow",
        "D   Cumulative predicted fuel burn",
    ]
    ylabels = ["Altitude (ft)", "Mach number", "Fuel flow (kg/s)", "Fuel burn (kg)"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, loc="left", pad=10, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max_time * 1.07)
        ax.grid(axis="y", color="#D9E0E5", lw=0.6)
        ax.set_axisbelow(True)
    low, high = exp["altitude_bounds_ft"]
    axes[0, 0].set_ylim(low - 500, high + 500)
    axes[0, 0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    min_mach = min(item["mach"] for item in report["solves"].values())
    axes[0, 1].set_ylim(min_mach - 0.02, 0.837)
    axes[0, 1].axhline(0.82, color="#89939A", lw=0.8, ls=":", zorder=0)
    axes[0, 1].text(0.4, 0.8235, "Aircraft limit: M 0.82", color="#687781", fontsize=8)
    margin = max(0.01, (max(rates) - min(rates)) * 0.2)
    axes[1, 0].set_ylim(min(rates) - margin, max(rates) + margin)
    max_fuel = max(item["fuel_kg"] for item in report["solves"].values())
    axes[1, 1].set_ylim(0, max_fuel * 1.16)
    axes[1, 1].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax in axes[1]:
        ax.set_xlabel("Elapsed cruise time (min)")
    fig.text(
        0.11,
        0.065,
        f"Both runs use {baseline_label} force constraints "
        "and their respective fuel models.\n"
        "Single deterministic solves; differences do not establish real fuel savings.",
        fontsize=9,
        color="#536572",
        linespacing=1.6,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_bada4" if baseline == "bada4" else ""
    target = args.output_dir / f"neural_fuel_comparison{suffix}.png"
    fig.savefig(target, dpi=150, facecolor="white")
    print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
