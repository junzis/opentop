"""Uncalibrated wave-fuel correction sensitivity on the saved Mach sweep.

This changes no optimization model and performs no trajectory solves.
"""

import argparse
from pathlib import Path

import matplotlib

import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.sweep)
    data = data.loc[data.mach <= 0.82000001]
    wave_cost = data.openap_kg_km - data.no_wave_kg_km
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.78)
    fig.text(
        0.12,
        0.95,
        "What if we restore a wave-drag fuel penalty?",
        fontsize=15,
        weight="bold",
        color="#243746",
    )
    fig.text(
        0.12,
        0.90,
        "A320 | 66,300 kg | 32,000 ft | Fixed-state sensitivity only",
        color="#536572",
    )
    for alpha, color, line in [
        (0, "#D55E00", "--"),
        (0.5, "#CC79A7", ":"),
        (1, "#0072B2", "-"),
        (1.5, "#009E73", "-."),
    ]:
        cost = data.acropole_kg_km + alpha * wave_cost
        i = cost.idxmin()
        mach = float(data.loc[i, "mach"])
        ax.plot(
            data.mach,
            cost,
            color=color,
            ls=line,
            lw=2,
            label=f"Correction strength {alpha:g}  |  minimum at M {mach:.3f}",
        )
        ax.plot(mach, cost.loc[i], "o", color=color, mec="white", ms=7)
        print(f"alpha={alpha:g}: minimum Mach={mach:.4f}, cost={cost.loc[i]:.6f} kg/km")
    ax.set_xlim(0.70, 0.825)
    ax.set_ylim(2.90, 3.48)
    ax.axvline(0.82, color="#999999", ls=":", lw=0.8)
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Predicted fuel per distance (kg/km)")
    ax.grid(axis="y", color="#DDE3E8", lw=0.7)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(
        0.12,
        0.065,
        "Candidate = Acropole + strength x (OpenAP wave-on minus wave-off fuel).\n"
        "Uncalibrated correction; no claim of improved real-world accuracy.",
        fontsize=9,
        color="#536572",
        linespacing=1.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
