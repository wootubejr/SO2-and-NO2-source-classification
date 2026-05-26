"""
ERA5 850 hPa Wind → Parquet Preprocessor
==========================================
Reads u/v wind components at 850 hPa from the ARCO ERA5 Zarr store
on GCS, writes hourly Parquet files to your GCS bucket.

Why 850 hPa:
  - Represents ~1.5 km altitude — the boundary layer top where most
    industrial SO₂ is transported before mixing out
  - Volcanic SO₂ from passive degassing also concentrates here
    (major eruptions inject higher, but those are rare events)
  - Standard level used by Fioletov et al. (2023) for emission estimation

Why Parquet not Zarr:
  - Member 2 will join these wind fields with TROPOMI pixels in Spark
  - Spark reads Parquet natively with predicate pushdown
  - Spark cannot read Zarr without custom UDFs, which defeats the
    purpose of the two-stage architecture

Output schema (agreed with Member 2):
  - date (date32): partition column
  - hour (int8): 0–23 UTC
  - era5_lat (float32): native ERA5 grid latitude (0.25° spacing)
  - era5_lon (float32): native ERA5 grid longitude (0.25° spacing)
  - u_wind_850 (float32): u-component of wind at 850 hPa (m/s)
  - v_wind_850 (float32): v-component of wind at 850 hPa (m/s)

Member 2 will interpolate from ERA5's 0.25° grid to TROPOMI pixel
locations — do NOT pre-interpolate here. Keeping the native grid
lets Member 2 choose the interpolation method (nearest-neighbor
vs bilinear) and avoids locking in assumptions.

Usage:
  Local test (one day):
    python era5_to_parquet.py \
      --start-date 2023-06-15 --end-date 2023-06-15 \
      --output ./local_test/era5/

  Full run (GCP VM):
    python era5_to_parquet.py \
      --start-date 2023-06-01 --end-date 2023-09-30 \
      --output gs://your-bucket/era5/wind_850hpa/

Dependencies:
  pip install xarray zarr gcsfs pyarrow pandas numpy google-cloud-storage
"""

import argparse
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("era5_preprocess")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

ERA5_ZARR_PATH = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

PRESSURE_LEVEL = 850  # hPa

PARQUET_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("hour", pa.int8()),
    ("era5_lat", pa.float32()),
    ("era5_lon", pa.float32()),
    ("u_wind_850", pa.float32()),
    ("v_wind_850", pa.float32()),
])


# ═══════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_parquet(path: str, table: pa.Table):
    if path.startswith("gs://"):
        from google.cloud import storage
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        parts = path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        blob = client.bucket(parts[0]).blob(parts[1])
        blob.upload_from_file(buf, content_type="application/octet-stream")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path, compression="snappy")


def output_exists(path: str) -> bool:
    if path.startswith("gs://"):
        from google.cloud import storage
        parts = path.replace("gs://", "").split("/", 1)
        return storage.Client().bucket(parts[0]).blob(parts[1]).exists()
    return os.path.exists(path)


# ═══════════════════════════════════════════════════════════════════
# ERA5 EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def open_era5() -> xr.Dataset:
    """
    Open the ARCO ERA5 Zarr store lazily.
    No data is downloaded until .compute() or .values is called.
    The Zarr chunking means only the requested time/level slices
    are fetched from GCS.
    """
    ds = xr.open_zarr(
        ERA5_ZARR_PATH,
        chunks={"time": 24},  # one day at a time
        storage_options=dict(token="anon"),
    )
    return ds


def extract_one_day(
    ds: xr.Dataset, target_date: datetime, output_base: str
) -> dict:
    """
    Extract u/v wind at 850 hPa for all 24 hours of one day.
    Writes one Parquet file per day.

    Returns stats dict.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    out_path = f"{output_base}/{date_str}.parquet"

    stats = {"date": date_str, "status": "skip", "rows": 0}

    if output_exists(out_path):
        stats["status"] = "exists"
        return stats

    try:
        # Select one day + 850 hPa
        # ERA5 ARCO uses 'level' for pressure levels
        day_start = f"{date_str}T00:00:00"
        day_end = f"{date_str}T23:00:00"

        slice_ds = ds[["u_component_of_wind", "v_component_of_wind"]].sel(
            level=PRESSURE_LEVEL,
            time=slice(day_start, day_end),
        )

        # .compute() triggers the actual download from GCS
        # For one day at one level, this is ~721 × 1440 × 24 × 2 floats
        # ≈ 190 MB uncompressed — fits easily in memory
        data = slice_ds.compute()

        # Flatten to tabular format
        rows = []
        for t_idx, t_val in enumerate(data.time.values):
            ts = pd.Timestamp(t_val)
            hour = ts.hour

            u = data["u_component_of_wind"].isel(time=t_idx).values
            v = data["v_component_of_wind"].isel(time=t_idx).values
            lats = data.latitude.values
            lons = data.longitude.values

            # Create meshgrid and flatten
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            df_hour = pd.DataFrame({
                "date": ts.date(),
                "hour": np.int8(hour),
                "era5_lat": lat_grid.flatten().astype(np.float32),
                "era5_lon": lon_grid.flatten().astype(np.float32),
                "u_wind_850": u.flatten().astype(np.float32),
                "v_wind_850": v.flatten().astype(np.float32),
            })
            rows.append(df_hour)

        df_day = pd.concat(rows, ignore_index=True)

        # Write
        table = pa.Table.from_pandas(df_day, schema=PARQUET_SCHEMA,
                                     preserve_index=False)
        write_parquet(out_path, table)

        stats["status"] = "ok"
        stats["rows"] = len(df_day)

    except Exception as e:
        stats["status"] = f"error: {e}"
        log.warning(f"Failed for {date_str}: {e}")

    return stats


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_era5_output(output_base: str, sample_date: str):
    """Quick validation of one day's ERA5 Parquet output."""
    path = f"{output_base}/{sample_date}.parquet"

    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        table = pq.read_table(path, filesystem=fs)
    else:
        table = pq.read_table(path)

    df = table.to_pandas()
    print(f"=== ERA5 VALIDATION: {sample_date} ===")
    print(f"Schema: {table.schema}")
    print(f"Rows: {len(df):,}")
    print(f"Hours present: {sorted(df['hour'].unique())}")
    print(f"Lat range: [{df['era5_lat'].min():.2f}, {df['era5_lat'].max():.2f}]")
    print(f"Lon range: [{df['era5_lon'].min():.2f}, {df['era5_lon'].max():.2f}]")
    print(f"u_wind range: [{df['u_wind_850'].min():.2f}, {df['u_wind_850'].max():.2f}] m/s")
    print(f"v_wind range: [{df['v_wind_850'].min():.2f}, {df['v_wind_850'].max():.2f}] m/s")

    # Expected: 721 lats × 1440 lons × 24 hours = ~24.9M rows per day
    expected_rows = 721 * 1440 * 24
    print(f"Expected rows: {expected_rows:,}")
    assert len(df) == expected_rows, f"FAIL: expected {expected_rows}, got {len(df)}"
    assert len(df["hour"].unique()) == 24, "FAIL: missing hours"
    assert df["u_wind_850"].notna().all(), "FAIL: NaN in u_wind"
    assert df["v_wind_850"].notna().all(), "FAIL: NaN in v_wind"
    print("✓ All validation checks passed")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract ERA5 850 hPa wind to Parquet"
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=True,
                        help="Local path or gs://bucket/prefix")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.validate_only:
        validate_era5_output(args.output, args.start_date)
        return

    log.info(f"Opening ERA5 Zarr store (lazy)...")
    ds = open_era5()
    log.info(f"ERA5 variables: {list(ds.data_vars)}")

    total_ok = 0
    total_err = 0
    total_rows = 0

    current = start
    n_days = (end - start).days + 1
    day_num = 0

    while current <= end:
        day_num += 1
        stats = extract_one_day(ds, current, args.output)

        if stats["status"] == "ok":
            total_ok += 1
            total_rows += stats["rows"]
        elif stats["status"] == "exists":
            pass
        else:
            total_err += 1

        if day_num % 7 == 0:
            log.info(
                f"  Progress: {day_num}/{n_days} days | "
                f"OK={total_ok} | Rows={total_rows:,}"
            )
        current += timedelta(days=1)

    log.info("=" * 60)
    log.info("ERA5 PREPROCESSING COMPLETE")
    log.info(f"  Days processed: {total_ok}")
    log.info(f"  Errors: {total_err}")
    log.info(f"  Total rows: {total_rows:,}")
    log.info(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
