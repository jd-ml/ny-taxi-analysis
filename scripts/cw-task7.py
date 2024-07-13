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
    IntegerType,
)
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, unix_timestamp, from_unixtime, sum, max, lit, round, expr
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
# UDFs for task7, pickup time and passenger count
# --------------------------------------------------------------------------------------------------------------------- #

def fields_yellow(dataframe):
    dataframe = dataframe.select(
        dataframe["tpep_pickup_datetime"],
        dataframe["passenger_count"],
    )
    return dataframe

def fields_green(dataframe):
    dataframe = dataframe.select(
        dataframe["lpep_pickup_datetime"],
        dataframe["passenger_count"],
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
    yellow_tripdata_df = fieldcleansing(yellow_tripdata_df)
    green_tripdata_df = fieldcleansing(green_tripdata_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_tripdata_df)
    green_prefilter_df = fields_green(green_tripdata_df)

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Calculations
    # --------------------------------------------------------------------------------------------------------------------------------------------- #    
    
    # total trip counts for both dataframes
    yellow_tc = yellow_prefilter_df.count()
    green_tc = green_prefilter_df.count()
    
    # add total trip counts to blank dataframe
    yellow_counts_df = yellow_prefilter_df.withColumn('yellow_total', lit(yellow_tc))
    green_counts_df = green_prefilter_df.withColumn('green_total', lit(green_tc))
    
    # filter dataframes for less than 1 person
    yellow_singles = yellow_prefilter_df[(yellow_prefilter_df['passenger_count'] == '1.0')]
    green_singles = green_prefilter_df[(green_prefilter_df['passenger_count'] == '1.0')]

    # total trip counts for both dataframes
    yellow_single_tc = yellow_singles.count()
    green_single_tc = green_singles.count()
    
    # add to dataframe
    yellow_counts_df = yellow_counts_df.withColumn('yellow_single_total', lit(yellow_single_tc))
    green_counts_df = green_counts_df.withColumn('green_single_total', lit(green_single_tc))

    # calculate percentages and add column
    yellow_counts_df = yellow_counts_df.withColumn('percentage', lit((yellow_single_tc / yellow_tc) * 100))
    green_counts_df = green_counts_df.withColumn('percentage', lit((green_single_tc / green_tc) * 100))

    # drop columns
    yellow_counts_df = yellow_counts_df.drop('tpep_pickup_datetime', 'passenger_count')
    green_counts_df = green_counts_df.drop('lpep_pickup_datetime', 'passenger_count')

    # # get first row
    yellow_counts_df = yellow_counts_df.limit(1)
    green_counts_df = green_counts_df.limit(1)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    yellow_counts_df = yellow_counts_df.coalesce(1)
    green_counts_df = green_counts_df.coalesce(1)
    
    # write dataframes out to files in .csv format
    yellow_counts_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t7_yellow_percentage_" + date_time)
    green_counts_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t7_green_percentage_" + date_time)
    
    spark.stop()    