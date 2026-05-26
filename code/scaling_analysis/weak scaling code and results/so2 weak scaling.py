#!/usr/bin/env python
"""
 Weak Scaling Experiment

Experiment design:
  2 workers  + 1 day  of data  
  4 workers  + 2 days of data  
  8 workers  + 4 days of data  
  16 workers + 8 days of data 
"""

import argparse
import json
import math
import time
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, FloatType, StringType, IntegerType
)
from pyspark.sql.functions import pandas_udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

ALL_DAYS = [
    "2023-06-01", "2023-06-02", "2023-06-03", "2023-06-04",
    "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08",
]

BUCKET          = "gs://st446-so2-data"
FIOLETOV        = f"{BUCKET}/fioletov_catalogue.csv"
OUTPUT_BASE     = f"{BUCKET}/weak_scaling_results"

# Pipeline constants
MAX_LABEL_DIST_KM     = 150.0
R_EARTH_M             = 6371000.0
SO2_RESIDENCE_SECONDS = 86400
DEG_PER_METER_LAT     = 1.0 / (R_EARTH_M * math.pi / 180.0)

# GBT hyperparameters
GBT_MAX_ITER  = 100
GBT_MAX_DEPTH = 5
GBT_STEP_SIZE = 0.1
GBT_SUBSAMPLE = 0.8


def tick(label, timings):
    timings[label] = {"start": time.time()}
    print(f"\n{'='*60}")
    print(f"STAGE START: {label}")
    print(f"{'='*60}")


def tock(label, timings):
    elapsed = time.time() - timings[label]["start"]
    timings[label]["elapsed_s"] = round(elapsed, 2)
    print(f"STAGE END:   {label}  →  {elapsed:.1f}s")
    return elapsed


def build_paths(days):
    """Build GCS glob paths for the given list of date strings."""
    so2_paths  = [f"{BUCKET}/tropomi/SO2/{d}/*.parquet" for d in days]
    no2_paths  = [f"{BUCKET}/tropomi/NO2/{d}/*.parquet" for d in days]
    # ERA5 files are named by date e.g. 2023-06-01.parquet
    era5_paths = [f"{BUCKET}/era5/era5/wind_850hpa/{d}.parquet" for d in days]
    return so2_paths, no2_paths, era5_paths


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True,
                        help="Number of worker nodes")
    parser.add_argument("--days", type=int, required=True,
                        help="Number of days of data to process (1, 2, or 4)")
    args = parser.parse_args()

    n_workers = args.workers
    n_days    = args.days

    # Select first n_days from available days
    selected_days = ALL_DAYS[:n_days]
    print(f"Workers: {n_workers}, Days: {n_days}")
    print(f"Selected days: {selected_days}")

    so2_paths, no2_paths, era5_paths = build_paths(selected_days)

    timings = {}
    metrics = {}

    # ── Spark Session ───────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName(f"SO2_WeakScaling_{n_workers}w_{n_days}d")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.shuffle.partitions", str(n_workers * 50))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # stage1 load data
    tick("stage1_data_load", timings)

    so2  = spark.read.parquet(*so2_paths)
    no2  = spark.read.parquet(*no2_paths)
    era5 = spark.read.parquet(*era5_paths)

    so2_count  = so2.count()
    no2_count  = no2.count()
    era5_count = era5.count()

    print(f"SO2  rows: {so2_count:,}")
    print(f"NO2  rows: {no2_count:,}")
    print(f"ERA5 rows: {era5_count:,}")

    metrics["so2_input_rows"]  = so2_count
    metrics["no2_input_rows"]  = no2_count
    metrics["era5_input_rows"] = era5_count

    tock("stage1_data_load", timings)

    # stage 2 spatial join
    tick("stage2_colocation", timings)

    so2_snapped = so2.withColumn(
        "lat_key", (F.round(F.col("lat") / 0.1) * 0.1).cast("float")
    ).withColumn(
        "lon_key", (F.round(F.col("lon") / 0.1) * 0.1).cast("float")
    )

    no2_snapped = no2.withColumn(
        "lat_key", (F.round(F.col("lat") / 0.1) * 0.1).cast("float")
    ).withColumn(
        "lon_key", (F.round(F.col("lon") / 0.1) * 0.1).cast("float")
    ).select("date", "lat_key", "lon_key", F.col("vcd").alias("no2_vcd"))

    coloc = so2_snapped.join(
        no2_snapped, on=["date", "lat_key", "lon_key"], how="left"
    ).drop("lat_key", "lon_key")

    COLOC_CHECKPOINT = f"{OUTPUT_BASE}/checkpoints/coloc_{n_workers}w_{n_days}d.parquet"
    coloc.write.mode("overwrite").parquet(COLOC_CHECKPOINT)
    coloc = spark.read.parquet(COLOC_CHECKPOINT)

    coloc_count = coloc.count()
    print(f"Co-located pixel pairs: {coloc_count:,}")
    metrics["coloc_rows"] = coloc_count

    tock("stage2_colocation", timings)

    # stage 3: Join ERA5 winds
    tick("stage3_era5_join", timings)

    coloc = coloc.withColumn(
        "tropomi_lon_360", ((F.col("lon") + 360) % 360)
    ).withColumn(
        "era5_lat_key", (F.round(F.col("lat") / 0.25) * 0.25).cast("float")
    ).withColumn(
        "era5_lon_key", (F.round(F.col("tropomi_lon_360") / 0.25) * 0.25).cast("float")
    ).withColumn(
        "hour", F.hour(F.col("time"))
    )

    era5_keyed = era5.withColumn(
        "era5_lat_key", F.col("era5_lat").cast("float")
    ).withColumn(
        "era5_lon_key", F.col("era5_lon").cast("float")
    )

    coloc = coloc.join(
        era5_keyed.select(
            "date", "hour", "era5_lat_key", "era5_lon_key",
            "u_wind_850", "v_wind_850"
        ),
        on=["date", "hour", "era5_lat_key", "era5_lon_key"],
        how="left"
    ).withColumn(
        "wind_speed",
        F.sqrt(F.col("u_wind_850")**2 + F.col("v_wind_850")**2)
    )

    era5_joined_count = coloc.count()
    print(f"Rows after ERA5 join: {era5_joined_count:,}")
    metrics["era5_joined_rows"] = era5_joined_count

    tock("stage3_era5_join", timings)

    # stage 4 Wind Back-Trajectory, Fioletov Labelling
    tick("stage4_labelling", timings)

    coloc = coloc.withColumn(
        "source_lat",
        F.col("lat") - (
            F.col("v_wind_850") * SO2_RESIDENCE_SECONDS * DEG_PER_METER_LAT
        )
    ).withColumn(
        "source_lon",
        F.col("lon") - (
            F.col("u_wind_850") * SO2_RESIDENCE_SECONDS * DEG_PER_METER_LAT /
            F.cos(F.col("lat") * math.pi / 180.0)
        )
    ).withColumn(
        "source_lat",
        F.greatest(F.lit(-90.0), F.least(F.lit(90.0), F.col("source_lat")))
    )

    fioletov_pd = pd.read_csv(FIOLETOV)
    fioletov_bc = spark.sparkContext.broadcast(fioletov_pd)

    result_schema = StructType([
        StructField("nearest_source_name", StringType()),
        StructField("source_type", StringType()),
        StructField("country", StringType()),
        StructField("wind_corrected_dist_km", FloatType()),
        StructField("source_elevation_m", IntegerType()),
        StructField("source_amf", FloatType()),
    ])

    @pandas_udf(result_schema)
    def nearest_source_udf(
        source_lats: pd.Series, source_lons: pd.Series
    ) -> pd.DataFrame:
        cat = fioletov_bc.value
        cat_lats = np.radians(cat["latitude"].values)
        cat_lons = np.radians(cat["longitude"].values)
        slats = np.radians(source_lats.values)[:, np.newaxis]
        slons = np.radians(source_lons.values)[:, np.newaxis]
        dlat = cat_lats - slats
        dlon = cat_lons - slons
        a = (np.sin(dlat/2)**2 +
             np.cos(slats) * np.cos(cat_lats) * np.sin(dlon/2)**2)
        dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        idx = np.argmin(dist_km, axis=1)
        min_dists = dist_km[np.arange(len(idx)), idx]
        return pd.DataFrame({
            "nearest_source_name": cat["source_name"].iloc[idx].values,
            "source_type":         cat["source_type"].iloc[idx].values,
            "country":             cat["country"].iloc[idx].values,
            "wind_corrected_dist_km": min_dists.astype(np.float32),
            "source_elevation_m":  cat["elevation_m"].iloc[idx].values.astype(np.int32),
            "source_amf":          cat["amf"].iloc[idx].values.astype(np.float32),
        })

    coloc = coloc.withColumn(
        "nearest", nearest_source_udf(F.col("source_lat"), F.col("source_lon"))
    ).select("*", "nearest.*").drop("nearest")

    coloc = coloc.filter(
        F.col("wind_corrected_dist_km") <= MAX_LABEL_DIST_KM
    ).withColumn(
        "label", F.when(F.col("source_type") == "Volcano", 1).otherwise(0)
    )

    coloc = coloc.withColumn(
        "so2_no2_ratio",
        F.col("vcd") / F.when(F.col("no2_vcd") != 0, F.col("no2_vcd")).otherwise(None)
    ).withColumn(
        "vcd_height_ratio",
        F.col("so2_vcd_7km") / F.when(
            F.col("so2_vcd_1km") != 0, F.col("so2_vcd_1km")
        ).otherwise(None)
    ).withColumn(
        "raw_dist_km",
        F.sqrt(
            ((F.col("lat") - F.col("source_lat")) * 111.0)**2 +
            ((F.col("lon") - F.col("source_lon")) * 111.0 *
             F.cos(F.col("lat") * math.pi / 180.0))**2
        )
    ).withColumn(
        "wind_x_ratio", F.col("wind_speed") * F.col("so2_no2_ratio")
    ).withColumn(
        "month", F.month(F.col("time"))
    ).withColumn(
        "hour_of_day", F.hour(F.col("time"))
    )

    feature_cols = [
        "so2_no2_ratio", "vcd_height_ratio", "aerosol_index",
        "wind_corrected_dist_km", "raw_dist_km", "source_elevation_m",
        "wind_speed", "wind_x_ratio", "cloud_fraction",
        "solar_zenith_angle", "sensor_zenith_angle", "surface_altitude",
        "month", "hour_of_day", "label"
    ]

    final = coloc.select(feature_cols).dropna(
        subset=["so2_no2_ratio", "vcd_height_ratio", "wind_corrected_dist_km", "label"]
    )

    labelled_count   = final.count()
    volcanic_count   = final.filter("label=1").count()
    industrial_count = final.filter("label=0").count()

    print(f"Labelled pixels:    {labelled_count:,}")
    print(f"Volcanic (label=1): {volcanic_count:,}")
    print(f"Industrial (label=0): {industrial_count:,}")

    metrics["labelled_rows"]   = labelled_count
    metrics["volcanic_rows"]   = volcanic_count
    metrics["industrial_rows"] = industrial_count

    tock("stage4_labelling", timings)

    # stage5 GBT Training and Evaluation
    tick("stage5_gbt", timings)

    total = labelled_count
    w_volcanic   = total / (2 * volcanic_count)   if volcanic_count   > 0 else 1.0
    w_industrial = total / (2 * industrial_count) if industrial_count > 0 else 1.0

    final = final.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, w_volcanic).otherwise(w_industrial)
    ).withColumn(
        "hour_sin", F.sin(F.col("hour_of_day") * 2 * math.pi / 24)
    ).withColumn(
        "hour_cos", F.cos(F.col("hour_of_day") * 2 * math.pi / 24)
    ).drop("hour_of_day", "month")

    ml_features = [
        "so2_no2_ratio", "vcd_height_ratio", "aerosol_index",
        "wind_corrected_dist_km", "raw_dist_km", "source_elevation_m",
        "wind_speed", "wind_x_ratio", "cloud_fraction",
        "solar_zenith_angle", "sensor_zenith_angle", "surface_altitude",
        "hour_sin", "hour_cos"
    ]

    train_v, test_v = final.filter(F.col("label") == 1).randomSplit([0.8, 0.2], seed=42)
    train_i, test_i = final.filter(F.col("label") == 0).randomSplit([0.8, 0.2], seed=42)
    train_df = train_v.union(train_i).cache()
    test_df  = test_v.union(test_i).cache()

    imputer   = Imputer(inputCols=["aerosol_index"], outputCols=["aerosol_index"], strategy="median")
    assembler = VectorAssembler(inputCols=ml_features, outputCol="features", handleInvalid="skip")
    gbt = GBTClassifier(
        labelCol="label", featuresCol="features", weightCol="class_weight",
        maxIter=GBT_MAX_ITER, maxDepth=GBT_MAX_DEPTH,
        stepSize=GBT_STEP_SIZE, subsamplingRate=GBT_SUBSAMPLE, seed=42
    )
    pipeline = Pipeline(stages=[imputer, assembler, gbt])

    fitted = pipeline.fit(train_df)
    preds  = fitted.transform(test_df).cache()

    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(preds)

    print(f"AUC-ROC:  {auc:.4f}")
    print(f"F1:       {f1:.4f}")

    metrics["auc_roc"] = round(auc, 4)
    metrics["f1"]      = round(f1, 4)

    tock("stage5_gbt", timings)

    # save results
    total_elapsed = sum(v["elapsed_s"] for v in timings.values())

    results = {
        "experiment":     "weak_scaling",
        "n_workers":      n_workers,
        "n_days":         n_days,
        "total_elapsed_s": round(total_elapsed, 2),
        "timings":        {k: v["elapsed_s"] for k, v in timings.items()},
        "metrics":        metrics,
    }

    print("\n" + "="*60)
    print("WEAK SCALING RUN COMPLETE")
    print(f"Workers: {n_workers} | Days: {n_days}")
    print(f"Total time: {total_elapsed:.1f}s")
    print("="*60)

    spark.sparkContext.parallelize([json.dumps(results, indent=2)], 1).saveAsTextFile(
        f"{OUTPUT_BASE}/run_{n_workers}w_{n_days}d_json"
    )

    spark.stop()


if __name__ == "__main__":
    main()
