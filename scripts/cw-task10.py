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
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, unix_timestamp, from_unixtime, sum, max, lit, round, udf, regexp_replace, trim, substring, expr, desc, asc, date_format

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

# UDFs for task9
def fields_yellow(dataframe):
    dataframe = dataframe.select(dataframe['tpep_pickup_datetime'])
    return dataframe

def fields_green(dataframe):
    dataframe = dataframe.select(dataframe['lpep_pickup_datetime'])
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
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-01.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-02.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-03.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-04.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-05.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-06.csv",
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/yellow_tripdata/2023/yellow_tripdata_2023-07.csv"
    ])

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

    # checking and removing any null values or wrong format in the dataset and cleaning them for further processing
    yellow_taxi_df = fieldcleansing(yellow_tripdata_df)
    green_taxi_df = fieldcleansing(green_tripdata_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_taxi_df)
    green_prefilter_df = fields_green(green_taxi_df)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Calculations
    # --------------------------------------------------------------------------------------------------------------------------------------------- #    
    
    # grouped dataframes and pickup counts
    yellow_pickups = yellow_prefilter_df.select(date_format('tpep_pickup_datetime','yyyy-MM').alias('month')).groupby('month').count()
    green_pickups = green_prefilter_df.select(date_format('lpep_pickup_datetime','yyyy-MM').alias('month')).groupby('month').count()

    # order them both (descending order)
    yellow_ordered_df = yellow_pickups.orderBy(desc("count"))
    green_ordered_df = green_pickups.orderBy(desc("count"))

    yellow_ordered_df.show()
    green_ordered_df.show()
    
    # filter for correct year
    top_5_yellow = yellow_ordered_df.limit(7)
    top_5_green = green_ordered_df.limit(7)

    # alter yellow names
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-01', 'January (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-02', 'February (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-03', 'March (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-04', 'April (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-05', 'May (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-06', 'June (Yellow)'))
    top_5_yellow = top_5_yellow.withColumn('month', regexp_replace('month', '2023-07', 'July (Yellow)'))

    # alter green names
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-01', 'January (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-02', 'February (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-03', 'March (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-04', 'April (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-05', 'May (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-06', 'June (Green)'))
    top_5_green = top_5_green.withColumn('month', regexp_replace('month', '2023-07', 'July (Green)'))

    # get top each
    top_5_yellow = top_5_yellow.limit(1)
    top_5_green = top_5_green.limit(1)
    
    # merge both tables
    grouped_df = top_5_yellow.union(top_5_green)

    # show grouped ascending
    grouped_df.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    grouped_df = grouped_df.coalesce(1)

    # write dataframes out to files in .csv format
    grouped_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t10_topmonths_" + date_time)

    spark.stop()      