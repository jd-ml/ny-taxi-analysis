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
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, unix_timestamp, from_unixtime, sum, max, lit, round, udf, regexp_replace, trim, substring, expr, desc, asc

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
    dataframe = dataframe.select(
        dataframe['tpep_pickup_datetime'],
        dataframe['Pickup_Borough']
    )
    return dataframe

def fields_green(dataframe):
    dataframe = dataframe.select(
        dataframe['lpep_pickup_datetime'],
        dataframe['Pickup_Borough']    
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
    yellow_taxi_df = fieldcleansing(yellow_tripdata_df)
    green_taxi_df = fieldcleansing(green_tripdata_df)

    # read taxi pick up data
    taxi_zone_lookup = spark.read.format("csv").option("header", True).option("inferSchema", True).load(
        "s3a://" + s3_data_repository_bucket + "/ECS765/nyc_taxi/taxi_zone_lookup.csv")
    
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

    # 1st green join w. PULocationID as key
    green_temp_df = green_tripdata_df.join(taxi_zone_lookup, green_tripdata_df["PULocationID"] == taxi_zone_lookup["LocationID"], "left_outer") \
        .select(green_tripdata_df["*"], 
                taxi_zone_lookup["Borough"].alias("Pickup_Borough"), 
                taxi_zone_lookup["Zone"].alias("Pickup_Zone"), 
                taxi_zone_lookup["service_zone"].alias("Pickup_service_zone")) \
        .drop("LocationID")


    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Selecting columns from dataframes
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    yellow_prefilter_df = fields_yellow(yellow_temp_df)
    green_prefilter_df = fields_green(green_temp_df)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Calculations
    # --------------------------------------------------------------------------------------------------------------------------------------------- #    
    
    # grouped dataframes and pickup counts
    yellow_pickups = yellow_prefilter_df.groupBy("Pickup_Borough").count()
    green_pickups = green_prefilter_df.groupBy("Pickup_Borough").count()

    # order them both (descending order)
    yellow_ordered_df = yellow_pickups.orderBy(desc("count"))
    green_ordered_df = green_pickups.orderBy(desc("count"))

    # get top 5 each
    top_5_yellow = yellow_ordered_df.limit(5)
    top_5_green = green_ordered_df.limit(5)

    # alter yellow names
    top_5_yellow = top_5_yellow.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Manhattan', 'Manhattan_Yellow'))
    top_5_yellow = top_5_yellow.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Queens', 'Queens_Yellow'))
    top_5_yellow = top_5_yellow.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Unknown', 'Unknown_Yellow'))
    top_5_yellow = top_5_yellow.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Brooklyn', 'Brooklyn_Yellow'))
    top_5_yellow = top_5_yellow.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Bronx', 'Bronx_Yellow'))

    # alter green names
    top_5_green = top_5_green.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Manhattan', 'Manhattan_Green'))
    top_5_green = top_5_green.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Queens', 'Queens_Green'))
    top_5_green = top_5_green.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Unknown', 'Unknown_Green'))
    top_5_green = top_5_green.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Brooklyn', 'Brooklyn_Green'))
    top_5_green = top_5_green.withColumn('Pickup_Borough', regexp_replace('Pickup_Borough', 'Bronx', 'Bronx_Green'))

    # merge both tables
    grouped_df = top_5_yellow.union(top_5_green)

    # ascending order
    grouped_df = grouped_df.orderBy(asc("count"))

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
    grouped_df.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t9_top10_" + date_time)

    spark.stop()      