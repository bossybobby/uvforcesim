from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_group_summary(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(
            columns=[
                "xr",
                "yr",
                "n_images",
                "zr_min",
                "zr_max",
                "abs_fz_min",
                "abs_fz_max",
                "z0_abs_xy",
            ]
        )

    summary = (
        records_df.assign(abs_fz=records_df["fz_mean"].abs())
        .groupby(["xr", "yr"], as_index=False)
        .agg(
            n_images=("image_path", "count"),
            zr_min=("zr", "min"),
            zr_max=("zr", "max"),
            abs_fz_min=("abs_fz", "min"),
            abs_fz_max=("abs_fz", "max"),
            z0_abs_xy=("z0_abs_xy", "first"),
        )
        .sort_values(["xr", "yr"])
        .reset_index(drop=True)
    )
    return summary


def plot_nominal_zero_hist(records_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    z0_values = records_df["z0_abs_xy"] if not records_df.empty else pd.Series(dtype=float)
    ax.hist(z0_values, bins=20, color="#4169e1", alpha=0.8, edgecolor="white")
    ax.set_title("Nominal zero distribution (z0_abs_xy)")
    ax.set_xlabel("z0_abs_xy (mm)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
