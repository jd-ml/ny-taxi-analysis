import sys, string
import os
import socket
import time
import operator
import boto3
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import max
from datetime import datetime
from pyspark.sql.types import (
    ShortType,
    StringType,
    StructType,
    StructField,
    TimestampType,
)
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.functions import to_date, count, col, approx_count_distinct
from graphframes import *

def fieldcleansing(dataframe):

  if (dataframe.first()['taxi_type']=='yellow_taxi'):
    columns = dataframe.columns
    dataframe = dataframe.select(*(col(c).cast("string").alias(c) for c in columns))
    dataframe = dataframe.filter(unix_timestamp(dataframe['tpep_pickup_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    dataframe = dataframe.filter(unix_timestamp(dataframe['tpep_dropoff_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    
  elif (dataframe.first()['taxi_type']=='green_taxi'):
    columns = dataframe.columns
    dataframe = dataframe.select(*(col(c).cast("string").alias(c) for c in columns))
    dataframe = dataframe.filter(unix_timestamp(dataframe['lpep_pickup_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    dataframe = dataframe.filter(unix_timestamp(dataframe['lpep_dropoff_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
  
  dataframe = dataframe.filter((dataframe["trip_distance"] >= 0) & (dataframe["fare_amount"] >= 0))
  return dataframe

# --------------------------------------------------------------------------------------------------------------------- #

# UDFs that extracts fare, distance and pickup date fields.
# fields are then filtered to exlude trips that have:
#         - a fare greater than $50
#         - distance less than 1 mile
#         - each day of first week of 2023

def fields_yellow(dataframe):
    dataframe = dataframe.select(
        dataframe["fare_amount"],
        dataframe["trip_distance"],
        dataframe["tpep_pickup_datetime"]
    )
    return dataframe

def fields_green(dataframe):
    dataframe = dataframe.select(
        dataframe["fare_amount"],
        dataframe["trip_distance"],
        dataframe["lpep_pickup_datetime"]
    )
    return dataframe

# --------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("TestDataset")\
        .getOrCreate()
    
    # shared read-only object bucket containing datasets
    s3_data_repository_bucket = os.environ['DATA_REPOSITORY_BUCKET']

    s3_endpoint_url = os.environ['S3_ENDPOINT_URL']+':'+os.environ['BUCKET_PORT']
    s3_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3_bucket = os.environ['BUCKET_NAME']

    hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoopConf.set("fs.s3a.endpoint", s3_endpoint_url)
    hadoopConf.set("fs.s3a.access.key", s3_access_key_id)
    hadoopConf.set("fs.s3a.secret.key", s3_secret_access_key)
    hadoopConf.set("fs.s3a.path.style.access", "true")
    hadoopConf.set("fs.s3a.connection.ssl.enabled", "false")

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # TASK 1
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # create dataframe for all yellow taxi data
    yellow_tripdata_df = spark.read.format("csv").option("header", True).option("inferSchema", True).load([
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-01.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-02.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-03.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-04.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-05.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-06.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-07.csv"
    ])
    yellow_tripdata_df.show()

    # create dataframe for all green taxi data
    green_tripdata_df = spark.read.format("csv").option("header", True).option("inferSchema", True).load([
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-01.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-02.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-03.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-04.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-05.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-06.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/green_tripdata/2023/green_tripdata_2023-07.csv",
    ])
    green_tripdata_df.show()

    # read taxi pick up data
    taxi_zone_lookup = spark.read.format("csv").option("header", True).option("inferSchema", True).load(
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/taxi_zone_lookup.csv")
    taxi_zone_lookup.show()
    
    # checking and removing any null values or wrong format in the dataset and cleaning them for further processing
    yellow_tripdata_df = fieldcleansing(yellow_tripdata_df)
    green_tripdata_df  =  fieldcleansing(green_tripdata_df)

    # 1st yellow join w. PULocationID as key
    yellow_temp_df = yellow_tripdata_df.join(taxi_zone_lookup, yellow_tripdata_df["PULocationID"] == taxi_zone_lookup["LocationID"], "left_outer") \
    .select(yellow_tripdata_df["*"], 
            taxi_zone_lookup["Borough"].alias("Pickup_Borough"), 
            taxi_zone_lookup["Zone"].alias("Pickup_Zone"), 
            taxi_zone_lookup["service_zone"].alias("Pickup_service_zone")) \
    .drop("LocationID")

    # 2nd yellow join w. DOLocationID as key
    yellow_taxi_df = yellow_temp_df.join(taxi_zone_lookup, yellow_temp_df["DOLocationID"] == taxi_zone_lookup["LocationID"], "left_outer") \
        .select(yellow_temp_df["*"], 
                taxi_zone_lookup["Borough"].alias("Dropoff_Borough"), 
                taxi_zone_lookup["Zone"].alias("Dropoff_Zone"), 
                taxi_zone_lookup["service_zone"].alias("Dropoff_service_zone")) \
        .drop("LocationID")

    # 1st green join w. PULocationID as key
    green_temp_df = green_tripdata_df.join(taxi_zone_lookup, green_tripdata_df["PULocationID"] == taxi_zone_lookup["LocationID"], "left_outer") \
        .select(green_tripdata_df["*"], 
                taxi_zone_lookup["Borough"].alias("Pickup_Borough"), 
                taxi_zone_lookup["Zone"].alias("Pickup_Zone"), 
                taxi_zone_lookup["service_zone"].alias("Pickup_service_zone")) \
        .drop("LocationID")

    # 2nd green join w. DOLocationID as key
    green_taxi_df = green_temp_df.join(taxi_zone_lookup, green_temp_df["DOLocationID"] == taxi_zone_lookup["LocationID"], "left_outer") \
        .select(green_temp_df["*"], 
                taxi_zone_lookup["Borough"].alias("Dropoff_Borough"), 
                taxi_zone_lookup["Zone"].alias("Dropoff_Zone"), 
                taxi_zone_lookup["service_zone"].alias("Dropoff_service_zone")) \
        .drop("LocationID")

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_taxi_df)
    green_prefilter_df = fields_green(green_taxi_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Filtering
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    
    # timestamp = datetime.strptime(datefield, "%Y-%m-%d %H:%M:%S")
    # timestamp = timestamp.strftime('%Y-%m-%d')
    
    yellow_postfilter_df = yellow_prefilter_df[
        (yellow_prefilter_df['tpep_pickup_datetime'] > '2023-01-01 00:00:00') \
        & (yellow_prefilter_df['tpep_pickup_datetime'] < '2023-01-07 23:59:59') \
        & (yellow_prefilter_df['fare_amount'] > '50') \
        & (yellow_prefilter_df['trip_distance'] < '1')
    ]
    
    green_postfilter_df = green_prefilter_df[
        (green_prefilter_df['lpep_pickup_datetime'] > '2023-01-01 00:00:00') \
        & (green_prefilter_df['lpep_pickup_datetime'] < '2023-01-07 23:59:59') \
        & (green_prefilter_df['fare_amount'] > '50') \
        & (green_prefilter_df['trip_distance'] < '1')
    ]

    yellow_postfilter_df = yellow_postfilter_df.withColumn("tpep_pickup_datetime", to_date(col('tpep_pickup_datetime')))
    green_postfilter_df = green_postfilter_df.withColumn("lpep_pickup_datetime", to_date(col('lpep_pickup_datetime')))

    yellow_postfilter_df.show()
    green_postfilter_df.show()
    
    yellow_weekone_df = yellow_postfilter_df.groupBy("tpep_pickup_datetime").count()
    green_weekone_df = green_postfilter_df.groupBy("lpep_pickup_datetime").count()

    yellow_weekone_df.show()
    green_weekone_df.show()

# --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    yellow_weekone_df = yellow_weekone_df.coalesce(1)
    green_weekone_df = green_weekone_df.coalesce(1)

    # write dataframes out to files in .csv format
    yellow_weekone_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t3_weekone_yellow" + date_time)
    green_weekone_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t3_weekone_green" + date_time)
    
    spark.stop()      