from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.uv_analysis.optical_flow import compute_dense_flow, extract_uv_features, load_gray


def compute_group_uv_features(group_df: pd.DataFrame, downsample: float = 0.5) -> pd.DataFrame:
    """Compute UV features from adjacent image pairs in a single XY group."""
    if len(group_df) < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    sorted_group = group_df.sort_values("zr", ascending=False).reset_index(drop=True)

    for i in range(len(sorted_group) - 1):
        prev_row = sorted_group.iloc[i]
        next_row = sorted_group.iloc[i + 1]

        prev_img = load_gray(prev_row["image_path"], downsample=downsample)
        next_img = load_gray(next_row["image_path"], downsample=downsample)

        u, v, magnitude = compute_dense_flow(prev_img, next_img)
        features = extract_uv_features(u, v, magnitude)
        rows.append(
            {
                "xr": float(next_row["xr"]),
                "yr": float(next_row["yr"]),
                "zr_prev": float(prev_row["zr"]),
                "zr_curr": float(next_row["zr"]),
                "zr_mid": float((prev_row["zr"] + next_row["zr"]) / 2.0),
                "fz_mean": float(next_row["fz_mean"]),
                "abs_fz_mean": float(abs(next_row["fz_mean"])),
                "z0_abs_xy": float(next_row["z0_abs_xy"]),
                "image_prev": str(prev_row["image_path"]),
                "image_curr": str(next_row["image_path"]),
                **features,
            }
        )

    return pd.DataFrame(rows)


def compute_all_groups_uv_features(
    grouped_records: dict[tuple[float, float], pd.DataFrame],
    downsample: float = 0.5,
) -> pd.DataFrame:
    frames = [compute_group_uv_features(group_df, downsample=downsample) for group_df in grouped_records.values()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
