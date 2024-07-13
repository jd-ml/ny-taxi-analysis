import sys, string
import os
import socket
import time
import operator
import boto3
import json
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.types import (
    ShortType,
    StringType,
    StructType,
    StructField,
    TimestampType,
)
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, unix_timestamp, from_unixtime, sum, max, lit, round
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

# UDF for task 5, to analyse trips to airport
def fields_yellow(dataframe):
    dataframe = dataframe.select(
        dataframe["airport_fee"],
        dataframe["tpep_pickup_datetime"]
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
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-07.csv"
    ])
    yellow_tripdata_df.show()

    # checking and removing any null values or wrong format in the dataset and cleaning them for further processing
    yellow_taxi_df = fieldcleansing(yellow_tripdata_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_taxi_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Filtering
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    # dataframe for the month of july, where there was a trip to the airport
    yellow_postfilter_df = yellow_prefilter_df[
        (yellow_prefilter_df['tpep_pickup_datetime'] > '2023-07-01 00:00:00') \
        & (yellow_prefilter_df['tpep_pickup_datetime'] < '2023-07-31 23:59:59') \
        & (yellow_prefilter_df['airport_fee'] > '0.0')
    ]

    # strip time off datetime
    yellow_postfilter_df = yellow_postfilter_df.withColumn("tpep_pickup_datetime", to_date(col('tpep_pickup_datetime')))

    # group and count by date
    yellow_march_df = yellow_postfilter_df.groupBy("tpep_pickup_datetime").count()
    yellow_march_df = yellow_march_df.orderBy("tpep_pickup_datetime")

    # calculation of total and rounded mean points
    yellow_march_df = yellow_march_df.withColumn("total", (lit(yellow_march_df.agg(sum(col('count'))).collect()[0][0]) / 31))
    yellow_march_df = yellow_march_df.select("*", round("total"))
    yellow_march_df = yellow_march_df.withColumnRenamed("round(total, 0)", "mean")
    yellow_march_df.show(truncate=False)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    yellow_march_df = yellow_march_df.coalesce(1)

    # write dataframes out to files in .csv format
    yellow_march_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t5_patterns_" + date_time)

    spark.stop()      