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
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, unix_timestamp, from_unixtime, sum, max, lit, round, udf, regexp_replace, trim, substring, expr
from graphframes import *

def fieldcleansing(dataframe):

  if (dataframe.first()['taxi_type']=='yellow_taxi'):
    columns = dataframe.columns
    dataframe = dataframe.select(*(col(c).cast("string").alias(c) for c in columns))
    dataframe = dataframe.filter(unix_timestamp(dataframe['tpep_pickup_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    dataframe = dataframe.filter(unix_timestamp(dataframe['tpep_dropoff_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    
    return dataframe
  elif (dataframe.first()['taxi_type']=='green_taxi'):
    columns = dataframe.columns
    dataframe = dataframe.select(*(col(c).cast("string").alias(c) for c in columns))
    dataframe = dataframe.filter(unix_timestamp(dataframe['lpep_pickup_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())
    dataframe = dataframe.filter(unix_timestamp(dataframe['lpep_dropoff_datetime'],'yyyy-MM-dd HH:mm:ss').isNotNull())

  return dataframe

# --------------------------------------------------------------------------------------------------------------------- #

# UDFs for task6, fare_amount, trip distance
def fields_yellow(dataframe):
    dataframe = dataframe.select(
        dataframe['tpep_pickup_datetime'],
        dataframe['tpep_dropoff_datetime'],
        dataframe['fare_amount']
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
    # Code from task 1
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # create dataframe for all yellow taxi data
    yellow_tripdata_df = spark.read.format("csv").option("header", True).option("inferSchema", True).load([
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-01.csv"
    ])

    # checking and removing any null values or wrong format in the dataset and cleaning them for further processing
    yellow_taxi_df = fieldcleansing(yellow_tripdata_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_taxi_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Filtering
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    # dataframe for the month of january, enforced by boolean condition
    yellow_jan_df = yellow_prefilter_df[
        (yellow_prefilter_df['tpep_pickup_datetime'] > '2023-01-01 00:00:00') \
        & (yellow_prefilter_df['tpep_pickup_datetime'] < '2023-01-31 23:59:59') \
    ]

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Calculations
    # --------------------------------------------------------------------------------------------------------------------------------------------- #    
    
    # add start time in string => timestamp format
    yellow_jan_df = yellow_jan_df.withColumn('start_ts', unix_timestamp(col('tpep_pickup_datetime'), "yyyy-MM-dd HH:mm:ss").cast("timestamp"))

    # add end time in string => timestamp format
    yellow_jan_df = yellow_jan_df.withColumn('end_ts', unix_timestamp(col('tpep_dropoff_datetime'), "yyyy-MM-dd HH:mm:ss").cast("timestamp"))

    # calculation
    yellow_jan_df = yellow_jan_df.withColumn('duration', expr("(end_ts - start_ts)"))

    # convert to string
    yellow_jan_df = yellow_jan_df.withColumn('duration', col("duration").cast(StringType()))
    
    # # regex strip
    # yellow_jan_df = yellow_jan_df.withColumn('duration_alt', regexp_replace("duration", r'[^0-9\.]', ""))

    # # rounded values
    # yellow_jan_df = yellow_jan_df.withColumn('duration_hours', substring('duration_alt', 1,5))

    # # drop all unnecessary columns
    # yellow_jan_df = yellow_jan_df.drop('tpep_pickup_datetime', 'tpep_dropoff_datetime', 'start_ts', 'end_ts', 'duration', 'duration_alt')

    # drop all unnecessary columns
    yellow_jan_df = yellow_jan_df.drop('tpep_pickup_datetime', 'tpep_dropoff_datetime', 'start_ts', 'end_ts')

    yellow_jan_df.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    yellow_jan_df = yellow_jan_df.coalesce(1)

    # write dataframes out to files in .csv format
    yellow_jan_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t8_temp_duration_" + date_time)

    spark.stop()      