#!/usr/bin/env python

"""
SO2 Emission Source Classification Member 4 Scaling Experiment

Runs the full end-to-end pipeline on TROPOMI satellite data to classify
SO2 emissions as volcanic or industrial in origin.

Pipeline stages:
  Stage 1 — Load TROPOMI SO2, NO2 and ERA5 wind data from GCS
  Stage 2 — Spatially co-locate SO2 and NO2 pixels on a 0.1-degree grid
  Stage 3 — Join ERA5 850hPa wind fields to each co-located pixel
  Stage 4 — Wind back-trajectory + Fioletov catalogue labelling
  Stage 5 — GBT classification (training + evaluation)

Stage timings and model metrics are written to GCS as JSON after each run
so results are preserved for the scaling analysis.
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

# GCS paths

SO2_PATH   = "gs://st446-so2-data/tropomi/SO2/2023-06-0[12345]/*.parquet"
NO2_PATH   = "gs://st446-so2-data/tropomi/NO2/2023-06-0[12345]/*.parquet"
ERA5_PATH  = "gs://st446-so2-data/era5/era5/wind_850hpa/*.parquet"
FIOLETOV   = "gs://st446-so2-data/fioletov_catalogue.csv"
OUTPUT_BASE = "gs://st446-so2-data/scaling_results"

# Physical constants
MAX_LABEL_DIST_KM     = 150.0
R_EARTH_M             = 6371000.0
SO2_RESIDENCE_SECONDS = 86400  # approximate SO2 atmospheric lifetime (~24h)
DEG_PER_METER_LAT     = 1.0 / (R_EARTH_M * math.pi / 180.0)

# GBT hyperparameters
GBT_MAX_ITER  = 100
GBT_MAX_DEPTH = 5
GBT_STEP_SIZE = 0.1
GBT_SUBSAMPLE = 0.8



def tick(label, timings):
    """Record the start time for a pipeline stage."""
    timings[label] = {"start": time.time()}
    print(f"\n{'='*60}")
    print(f"STAGE START: {label}")
    print(f"{'='*60}")


def tock(label, timings):
    """Record the end time and print elapsed seconds."""
    elapsed = time.time() - timings[label]["start"]
    timings[label]["elapsed_s"] = round(elapsed, 2)
    print(f"STAGE END:   {label}  →  {elapsed:.1f}s")
    return elapsed


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True,
                        help="Number of worker nodes (for labelling output only)")
    args = parser.parse_args()
    n_workers = args.workers

    timings = {}
    metrics = {}

    # Spark Session 
    spark = (
        SparkSession.builder
        .appName(f"SO2_Scaling_{n_workers}w")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"\nSpark version: {spark.version}")
    print(f"Workers configured: {n_workers}")

    #  Stage 1: Data Load 
    tick("stage1_data_load", timings)

    so2  = spark.read.parquet(SO2_PATH)
    no2  = spark.read.parquet(NO2_PATH)
    era5 = spark.read.parquet(ERA5_PATH)

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

    #  Stage 2: SO2/NO2 Spatial Co-location 
    # TROPOMI pixels are on an irregular swath grid, so both products
    # are snapped to a shared 0.1-degree grid before joining on date
    # and grid cell key.
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

    # Checkpoint to GCS to break the query plan lineage before the ERA5 join.
    # Without this, Spark tries to optimise across both joins simultaneously
    # which can produce very slow shuffle plans on large data.
    COLOC_CHECKPOINT = f"{OUTPUT_BASE}/checkpoints/coloc_{n_workers}w.parquet"
    coloc.write.mode("overwrite").parquet(COLOC_CHECKPOINT)
    coloc = spark.read.parquet(COLOC_CHECKPOINT)

    coloc_count = coloc.count()
    print(f"Co-located pixel pairs: {coloc_count:,}")
    metrics["coloc_rows"] = coloc_count

    tock("stage2_colocation", timings)

    # STAGE 3: ERA5 Wind Join 
    # ERA5 uses a 0-360 longitude convention while TROPOMI uses -180 to 180,
    # so longitudes are converted before snapping to the 0.25-degree ERA5 grid.
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
    missing_wind = coloc.filter("u_wind_850 is null").count()
    print(f"Rows after ERA5 join: {era5_joined_count:,}")
    print(f"Rows missing ERA5 wind: {missing_wind:,}")
    metrics["era5_joined_rows"] = era5_joined_count
    metrics["missing_wind_rows"] = missing_wind

    tock("stage3_era5_join", timings)

    #  Stage 4: Wind Back-Trajectory,  Fioletov Labelling 
    # Each pixel is projected back along the 850hPa wind vector for one
    # SO2 residence time to estimate where the emission originated.
    # The back-projected location is then matched against the Fioletov
    # catalogue using a vectorised Haversine UDF. Pixels within 150 km
    # of a catalogue entry are labelled; all others are discarded.
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

    # The catalogue has 759 rows so broadcasting it to every executor
    # avoids a shuffle and keeps this join fast
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
        """
        For each back-projected location, find the nearest entry in the
        Fioletov catalogue using a vectorised Haversine distance matrix.
        """
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

    # Feature engineering
    coloc = coloc.withColumn(
        # SO2/NO2 ratio is the primary discriminator: volcanoes emit
        # high SO2 but little NO2; industrial sources emit both
        "so2_no2_ratio",
        F.col("vcd") / F.when(F.col("no2_vcd") != 0, F.col("no2_vcd")).otherwise(None)
    ).withColumn(
        # Ratio of column retrievals at different altitude layers;
        # volcanic plumes tend to sit higher than surface industrial emissions
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

    labelled_count  = final.count()
    volcanic_count  = final.filter("label=1").count()
    industrial_count = final.filter("label=0").count()

    print(f"Labelled pixels:   {labelled_count:,}")
    print(f"Volcanic (label=1): {volcanic_count:,}")
    print(f"Industrial (label=0): {industrial_count:,}")

    metrics["labelled_rows"]   = labelled_count
    metrics["volcanic_rows"]   = volcanic_count
    metrics["industrial_rows"] = industrial_count

    tock("stage4_labelling", timings)

    #  STAGE 5: GBT Training and Inference
    tick("stage5_gbt", timings)

    # The dataset is imbalanced (~11% volcanic) so inverse-frequency
    # class weights are used to prevent the model from over-predicting
    # the industrial class
    total = labelled_count
    w_volcanic  = total / (2 * volcanic_count)  if volcanic_count  > 0 else 1.0
    w_industrial = total / (2 * industrial_count) if industrial_count > 0 else 1.0

    final = final.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, w_volcanic).otherwise(w_industrial)
    )

    # Encode hour cyclically so the model treats 23:00 and 00:00 as adjacent
    final = final.withColumn(
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

    # Stratified train/test split to preserve class balance in both sets
    train_v, test_v = final.filter(F.col("label") == 1).randomSplit([0.8, 0.2], seed=42)
    train_i, test_i = final.filter(F.col("label") == 0).randomSplit([0.8, 0.2], seed=42)
    train_df = train_v.union(train_i).cache()
    test_df  = test_v.union(test_i).cache()

    # Build pipeline: impute → assemble → GBT
    imputer = Imputer(
        inputCols=["aerosol_index"], outputCols=["aerosol_index"], strategy="median"
    )
    assembler = VectorAssembler(
        inputCols=ml_features, outputCol="features", handleInvalid="skip"
    )
    gbt = GBTClassifier(
        labelCol="label", featuresCol="features", weightCol="class_weight",
        maxIter=GBT_MAX_ITER, maxDepth=GBT_MAX_DEPTH,
        stepSize=GBT_STEP_SIZE, subsamplingRate=GBT_SUBSAMPLE, seed=42
    )
    pipeline = Pipeline(stages=[imputer, assembler, gbt])

    fitted = pipeline.fit(train_df)
    preds  = fitted.transform(test_df).cache()

    # Evaluate
    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(preds)
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)

    print(f"AUC-ROC:  {auc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    metrics["auc_roc"]  = round(auc, 4)
    metrics["f1"]       = round(f1, 4)
    metrics["accuracy"] = round(acc, 4)

    tock("stage5_gbt", timings)

    #  Save Results 
    total_elapsed = sum(v["elapsed_s"] for v in timings.values())

    results = {
        "n_workers": n_workers,
        "total_elapsed_s": round(total_elapsed, 2),
        "timings": {k: v["elapsed_s"] for k, v in timings.items()},
        "metrics": metrics,
    }

    print("\n" + "="*60)
    print("SCALING RUN COMPLETE")
    print(f"Workers:       {n_workers}")
    print(f"Total time:    {total_elapsed:.1f}s")
    print(f"Stage timings: {results['timings']}")
    print("="*60)

    # Write JSON result to GCS
    out_path = f"{OUTPUT_BASE}/run_{n_workers}w.json"
    results_json = json.dumps(results, indent=2)
    (
        spark.sparkContext
        .parallelize([results_json], 1)
        .saveAsTextFile(f"{OUTPUT_BASE}/run_{n_workers}w_json")
    )

    spark.stop()


if __name__ == "__main__":
    main()
