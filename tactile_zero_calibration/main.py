from __future__ import annotations

import argparse
from pathlib import Path

from src.data.exploration import build_group_summary, plot_nominal_zero_hist
from src.data.loader import group_by_xy, load_records, validate_unique_nominal_zero
from src.uv_analysis.pipeline import compute_all_groups_uv_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tactile true-zero calibration pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Input directory that contains PNG files")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument(
        "--uv_feature",
        type=str,
        default="uv_magnitude_max",
        help="UV feature for subsequent fitting (Phase 2)",
    )
    parser.add_argument("--fit_order", type=int, default=3, help="Polynomial order (Phase 2)")
    parser.add_argument("--downsample", type=float, default=0.5, help="Image downsample ratio for optical flow")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def run_phase1(data_dir: Path, output_dir: Path, downsample: float, verbose: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records_df = load_records(data_dir)
    records_out = output_dir / "records.parquet"
    if not records_df.empty:
        records_df.to_parquet(records_out, index=False)

    grouped = group_by_xy(records_df)
    violations = validate_unique_nominal_zero(grouped)

    summary_df = build_group_summary(records_df)
    summary_path = output_dir / "group_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    hist_path = output_dir / "nominal_zero_hist.png"
    plot_nominal_zero_hist(records_df, hist_path)

    uv_features_df = compute_all_groups_uv_features(grouped, downsample=downsample)
    uv_features_path = output_dir / "uv_features.csv"
    uv_features_df.to_csv(uv_features_path, index=False)

    report_path = output_dir / "phase1_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Phase 1 completed.\n")
        f.write(f"Total images: {len(records_df)}\n")
        f.write(f"Total XY groups: {len(grouped)}\n")
        f.write(f"Groups with non-unique z0_abs_xy: {len(violations)}\n")
        if violations:
            f.write("Violations (xy, unique_count):\n")
            for item in violations:
                f.write(f"  - {item}\n")

    if verbose:
        print(f"Loaded {len(records_df)} images from {data_dir}")
        print(f"Generated summary: {summary_path}")
        print(f"Generated histogram: {hist_path}")
        print(f"Generated UV features: {uv_features_path}")
        print(f"Generated report: {report_path}")


def main() -> None:
    args = parse_args()
    run_phase1(Path(args.data_dir), Path(args.output_dir), args.downsample, verbose=args.verbose)


if __name__ == "__main__":
    main()
