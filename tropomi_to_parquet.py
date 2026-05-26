"""

TROPOMI netCDF → Parquet Preprocessor (GCS output)
====================================================
Reads SO₂ and NO₂ L2 granules from the public S3 bucket,
writes quality-filtered Parquet to your GCS bucket.

Step 3 (local test):
  python tropomi_to_parquet.py \
    --start-date 2023-06-15 --end-date 2023-06-15 \
    --output ./local_test/ --workers 1 --products SO2

Step 5 (full run on GCP VM):
  python tropomi_to_parquet.py \
    --start-date 2023-06-01 --end-date 2023-09-30 \
    --output gs://your-bucket/tropomi/ --workers 16 --products SO2 NO2

Dependencies:
  pip install netcdf4 boto3 pyarrow pandas google-cloud-storage tqdm
"""

import argparse
import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("tropomi_preprocess")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

SOURCE_BUCKET = "meeo-s5p"
SOURCE_REGION = "eu-central-1"

PRODUCT_CONFIG = {
    "SO2": {
        "prefix": "OFFL/L2__SO2___",
        "column_var": "sulfurdioxide_total_vertical_column",
        "column_name": "so2_vcd",
        "qa_threshold": 0.5,
        "cloud_threshold": 0.3,
    },
    "NO2": {
        "prefix": "OFFL/L2__NO2___",
        "column_var": "nitrogendioxide_tropospheric_column",
        "column_name": "no2_vcd",
        "qa_threshold": 0.75,
        "cloud_threshold": 0.3,
    },
}

PARQUET_SCHEMA = pa.schema([
    ("lat", pa.float32()),
    ("lon", pa.float32()),
    ("time", pa.timestamp("s")),
    ("date", pa.date32()),
    ("orbit", pa.int32()),
    ("product", pa.string()),
    ("vcd", pa.float32()),
    ("qa_value", pa.float32()),
    ("cloud_fraction", pa.float32()),
    ("solar_zenith_angle", pa.float32()),
    ("sensor_zenith_angle", pa.float32()),
    ("so2_vcd_1km", pa.float32()),
    ("so2_vcd_7km", pa.float32()),
    ("aerosol_index", pa.float32()),
    ("surface_altitude", pa.float32()),
])


# ═══════════════════════════════════════════════════════════════════
# S3 INPUT (public, unsigned)
# ═══════════════════════════════════════════════════════════════════

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=SOURCE_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def list_granule_keys(product: str, start: datetime, end: datetime) -> List[str]:
    s3 = get_s3_client()
    prefix_base = PRODUCT_CONFIG[product]["prefix"]
    keys = []
    current = start
    while current <= end:
        prefix = f"{prefix_base}/{current:%Y/%m/%d}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=SOURCE_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".nc"):
                    keys.append(k)
        current += timedelta(days=1)
    return keys


# ═══════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════

def write_parquet_local(path: str, table: pa.Table):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def write_parquet_gcs(gcs_uri: str, table: pa.Table):
    from google.cloud import storage
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    blob = client.bucket(parts[0]).blob(parts[1])
    blob.upload_from_file(buf, content_type="application/octet-stream")


def write_parquet(path: str, table: pa.Table):
    if path.startswith("gs://"):
        write_parquet_gcs(path, table)
    else:
        write_parquet_local(path, table)


def output_exists(path: str) -> bool:
    if path.startswith("gs://"):
        from google.cloud import storage
        parts = path.replace("gs://", "").split("/", 1)
        return storage.Client().bucket(parts[0]).blob(parts[1]).exists()
    return os.path.exists(path)


# ═══════════════════════════════════════════════════════════════════
# NETCDF PARSING
# ═══════════════════════════════════════════════════════════════════

def extract_orbit(fname: str) -> int:
    m = re.search(r"_(\d{5})_\d{2}_\d{6}_", fname)
    return int(m.group(1)) if m else -1


def extract_timestamp(fname: str) -> Optional[datetime]:
    m = re.search(r"____(\d{8}T\d{6})_", fname)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    return None


def safe_get(ds, *path):
    """
    Navigate netCDF groups safely, returning a flattened numpy array.
    Returns None if any group/variable in the path doesn't exist.
    """
    node = ds
    for p in path:
        if hasattr(node, "groups") and p in node.groups:
            node = node.groups[p]
        elif hasattr(node, "variables") and p in node.variables:
            arr = node.variables[p][:]
            if hasattr(arr, "filled"):
                arr = arr.filled(np.nan)
            while arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr.flatten().astype(np.float32)
        else:
            return None
    return None


def first_valid(*arrays):
    """Return the first array that is not None. Replaces `or` chains
    which don't work on numpy arrays."""
    for arr in arrays:
        if arr is not None:
            return arr
    return None


def parse_granule(nc_bytes: bytes, key: str, product: str) -> Optional[pd.DataFrame]:
    import netCDF4

    cfg = PRODUCT_CONFIG[product]
    fname = key.split("/")[-1]
    orbit = extract_orbit(fname)
    ts = extract_timestamp(fname)
    if ts is None:
        return None

    try:
        ds = netCDF4.Dataset("in_memory.nc", memory=nc_bytes)
    except Exception as e:
        log.warning(f"Cannot open {fname}: {e}")
        return None

    try:
        # ── Core variables ──
        lat = safe_get(ds, "PRODUCT", "latitude")
        lon = safe_get(ds, "PRODUCT", "longitude")
        vcd = safe_get(ds, "PRODUCT", cfg["column_var"])
        qa  = safe_get(ds, "PRODUCT", "qa_value")

        if lat is None or lon is None or vcd is None or qa is None:
            log.warning(f"Missing core variables in {fname}")
            return None

        nan_fill = np.full(len(lat), np.nan, dtype=np.float32)

        # ── Optional variables with fallback paths ──
        cloud = first_valid(
            safe_get(ds, "PRODUCT", "SUPPORT_DATA", "INPUT_DATA",
                     "cloud_fraction_crb"),
            safe_get(ds, "PRODUCT", "SUPPORT_DATA", "INPUT_DATA",
                     "cloud_fraction"),
            nan_fill.copy(),
        )

        solar_zen = first_valid(
            safe_get(ds, "PRODUCT", "SUPPORT_DATA", "GEOLOCATIONS",
                     "solar_zenith_angle"),
            nan_fill.copy(),
        )

        sensor_zen = first_valid(
            safe_get(ds, "PRODUCT", "SUPPORT_DATA", "GEOLOCATIONS",
                     "viewing_zenith_angle"),
            safe_get(ds, "PRODUCT", "SUPPORT_DATA", "GEOLOCATIONS",
                     "sensor_zenith_angle"),
            nan_fill.copy(),
        )

        # ── Multi-height SO₂ retrievals (SO₂ only) ──
        if product == "SO2":
            so2_vcd_1km = first_valid(
                safe_get(ds, "PRODUCT", "SUPPORT_DATA", "DETAILED_RESULTS",
                         "sulfurdioxide_total_vertical_column_1km"),
                nan_fill.copy(),
            )
            so2_vcd_7km = first_valid(
                safe_get(ds, "PRODUCT", "SUPPORT_DATA", "DETAILED_RESULTS",
                         "sulfurdioxide_total_vertical_column_7km"),
                nan_fill.copy(),
            )
            aerosol_idx = first_valid(
                safe_get(ds, "PRODUCT", "SUPPORT_DATA", "INPUT_DATA",
                         "aerosol_index_340_380"),
                nan_fill.copy(),
            )
            surf_alt = first_valid(
                safe_get(ds, "PRODUCT", "SUPPORT_DATA", "INPUT_DATA",
                         "surface_altitude"),
                nan_fill.copy(),
            )
        else:
            so2_vcd_1km = nan_fill.copy()
            so2_vcd_7km = nan_fill.copy()
            aerosol_idx = nan_fill.copy()
            surf_alt = nan_fill.copy()

        # ── Quality filter ──
        mask = (
            (qa >= cfg["qa_threshold"])
            & (cloud < cfg["cloud_threshold"])
            & np.isfinite(vcd)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )
        n_good = int(mask.sum())
        if n_good == 0:
            return None

        # ── Build DataFrame ──
        df = pd.DataFrame({
            "lat": lat[mask],
            "lon": lon[mask],
            "time": pd.Timestamp(ts),
            "date": ts.date(),
            "orbit": np.int32(orbit),
            "product": product,
            "vcd": vcd[mask],
            "qa_value": qa[mask],
            "cloud_fraction": cloud[mask],
            "solar_zenith_angle": solar_zen[mask],
            "sensor_zenith_angle": sensor_zen[mask],
            "so2_vcd_1km": so2_vcd_1km[mask],
            "so2_vcd_7km": so2_vcd_7km[mask],
            "aerosol_index": aerosol_idx[mask],
            "surface_altitude": surf_alt[mask],
        })
        return df

    except Exception as e:
        log.warning(f"Parse error in {fname}: {e}")
        return None
    finally:
        ds.close()


# ═══════════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════════

def convert_one_granule(args: tuple) -> dict:
    key, product, output_base = args
    fname = key.split("/")[-1]
    ts = extract_timestamp(fname)
    date_str = ts.strftime("%Y-%m-%d") if ts else "unknown"

    out_name = fname.replace(".nc", ".parquet")
    out_path = f"{output_base}/{product}/{date_str}/{out_name}"

    stats = {"key": key, "status": "skip", "rows": 0,
             "bytes_in": 0, "bytes_out": 0}

    if output_exists(out_path):
        stats["status"] = "exists"
        return stats

    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=SOURCE_BUCKET, Key=key)
        nc_bytes = obj["Body"].read()
        stats["bytes_in"] = len(nc_bytes)
    except Exception as e:
        stats["status"] = f"download_error: {e}"
        return stats

    df = parse_granule(nc_bytes, key, product)
    del nc_bytes

    if df is None or len(df) == 0:
        stats["status"] = "empty_after_qa"
        return stats

    try:
        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA,
                                     preserve_index=False)
        write_parquet(out_path, table)
        stats["status"] = "ok"
        stats["rows"] = len(df)
        stats["bytes_out"] = table.nbytes
    except Exception as e:
        stats["status"] = f"write_error: {e}"

    return stats


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_output(output_base: str, product: str, sample_date: str):
    import pyarrow.parquet as pq

    path = f"{output_base}/{product}/{sample_date}/"
    if path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        files = fs.ls(path)
        if not files:
            print(f"ERROR: No files at {path}")
            return
        table = pq.read_table(files[0], filesystem=fs)
    else:
        import glob
        files = glob.glob(path + "*.parquet")
        if not files:
            print(f"ERROR: No files at {path}")
            return
        table = pq.read_table(files[0])

    df = table.to_pandas()
    print(f"=== VALIDATION: {product} {sample_date} ===")
    print(f"File: {files[0]}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nValue ranges:")
    for col in ["lat", "lon", "vcd", "qa_value", "cloud_fraction",
                "so2_vcd_1km", "so2_vcd_7km", "aerosol_index",
                "surface_altitude"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col:25s} min={df[col].min():>10.4f}  "
                  f"max={df[col].max():>10.4f}  "
                  f"non-null={non_null}/{len(df)}")
    print(f"\nOrbit: {sorted(df['orbit'].unique())}")
    print(f"Product: {df['product'].unique()}")

    assert df["lat"].between(-90, 90).all(), "FAIL: lat out of range"
    assert df["lon"].between(-180, 180).all(), "FAIL: lon out of range"
    assert (df["qa_value"] >= PRODUCT_CONFIG[product]["qa_threshold"]).all(), \
        "FAIL: qa_value filter not applied"
    assert (df["cloud_fraction"] < PRODUCT_CONFIG[product]["cloud_threshold"]).all(), \
        "FAIL: cloud_fraction filter not applied"

    if product == "SO2":
        n_1km = df["so2_vcd_1km"].notna().sum()
        n_7km = df["so2_vcd_7km"].notna().sum()
        print(f"\nMulti-height coverage: 1km={n_1km}, 7km={n_7km}")
        assert n_1km > 0, "FAIL: so2_vcd_1km all NaN"
        assert n_7km > 0, "FAIL: so2_vcd_7km all NaN"

    print("\n✓ All validation checks passed")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--products", nargs="+", default=["SO2", "NO2"],
                        choices=["SO2", "NO2"])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    if args.validate_only:
        for product in args.products:
            validate_output(args.output, product, args.start_date)
        return

    grand = {"total_granules": 0, "converted": 0, "empty": 0,
             "errors": 0, "skipped": 0, "total_rows": 0,
             "bytes_in": 0, "bytes_out": 0}

    for product in args.products:
        log.info(f"Listing {product}: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
        keys = list_granule_keys(product, start, end)
        log.info(f"Found {len(keys)} {product} granules")
        grand["total_granules"] += len(keys)
        if not keys:
            continue

        work = [(k, product, args.output) for k in keys]
        log.info(f"Converting with {args.workers} workers...")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one_granule, w): w for w in work}
            done = 0
            for future in as_completed(futures):
                stats = future.result()
                done += 1

                if stats["status"] == "ok":
                    grand["converted"] += 1
                    grand["total_rows"] += stats["rows"]
                    grand["bytes_in"] += stats["bytes_in"]
                    grand["bytes_out"] += stats["bytes_out"]
                elif stats["status"] == "exists":
                    grand["skipped"] += 1
                elif stats["status"] == "empty_after_qa":
                    grand["empty"] += 1
                else:
                    grand["errors"] += 1
                    log.warning(f"  FAIL: {stats['key']}: {stats['status']}")

                if done % 10 == 0:
                    log.info(f"  [{product}] {done}/{len(keys)} | "
                             f"OK={grand['converted']} | "
                             f"Rows={grand['total_rows']:,}")

    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info(f"  Total granules found:    {grand['total_granules']}")
    log.info(f"  Converted:               {grand['converted']}")
    log.info(f"  Empty (after QA filter): {grand['empty']}")
    log.info(f"  Errors:                  {grand['errors']}")
    log.info(f"  Skipped (already exist): {grand['skipped']}")
    log.info(f"  Total pixel rows:        {grand['total_rows']:,}")
    log.info(f"  Raw input size:          {grand['bytes_in']/1e9:.2f} GB")
    log.info(f"  Parquet output size:     {grand['bytes_out']/1e9:.2f} GB")
    if grand['bytes_in'] > 0:
        log.info(f"  Compression ratio:       "
                 f"{grand['bytes_out']/grand['bytes_in']*100:.1f}% of raw")
    log.info(f"  Output location:         {args.output}")


if __name__ == "__main__":
    main()
