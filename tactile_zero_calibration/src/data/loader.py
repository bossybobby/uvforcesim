from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<probe>[A-Za-z0-9\-]+)_"
    r"(?P<xr>-?\d+\.\d+)_"
    r"(?P<yr>-?\d+\.\d+)_"
    r"(?P<zr>-?\d+\.\d+)_"
    r"(?P<z0_abs_xy>-?\d+\.\d+)_"
    r"(?P<fx_mean>-?\d+\.\d+)_"
    r"(?P<fy_mean>-?\d+\.\d+)_"
    r"(?P<fz_mean>-?\d+\.\d+)\.png$"
)


@dataclass(frozen=True)
class ImageRecord:
    probe: str
    xr: float
    yr: float
    zr: float
    z0_abs_xy: float
    fx_mean: float
    fy_mean: float
    fz_mean: float
    image_path: Path


def parse_image_filename(file_path: Path) -> ImageRecord:
    """Parse one PNG filename into a strongly typed ImageRecord."""
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        raise ValueError(f"Unsupported filename format: {file_path.name}")

    values = match.groupdict()
    return ImageRecord(
        probe=values["probe"],
        xr=float(values["xr"]),
        yr=float(values["yr"]),
        zr=float(values["zr"]),
        z0_abs_xy=float(values["z0_abs_xy"]),
        fx_mean=float(values["fx_mean"]),
        fy_mean=float(values["fy_mean"]),
        fz_mean=float(values["fz_mean"]),
        image_path=file_path,
    )


def load_records(data_dir: str | Path) -> pd.DataFrame:
    """Load all PNG files from data_dir and parse filename metadata."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.png"))
    records = [parse_image_filename(file_path) for file_path in files]
    if not records:
        return pd.DataFrame(
            columns=[
                "probe",
                "xr",
                "yr",
                "zr",
                "z0_abs_xy",
                "fx_mean",
                "fy_mean",
                "fz_mean",
                "image_path",
            ]
        )

    frame = pd.DataFrame([record.__dict__ for record in records])
    return frame.sort_values(["xr", "yr", "zr"], ascending=[True, True, False]).reset_index(drop=True)


def group_by_xy(records_df: pd.DataFrame) -> dict[tuple[float, float], pd.DataFrame]:
    """Group records by (xr, yr) for per-point calibration processing."""
    if records_df.empty:
        return {}

    grouped = {}
    for (xr, yr), group in records_df.groupby(["xr", "yr"], sort=True):
        grouped[(float(xr), float(yr))] = group.sort_values("zr", ascending=False).reset_index(drop=True)
    return grouped
