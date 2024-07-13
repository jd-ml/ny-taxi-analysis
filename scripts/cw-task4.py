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
from pyspark.sql.functions import unix_timestamp
# import matplotlib.pyplot as plt
from pyspark.sql.functions import to_date, count, col, approx_count_distinct, desc
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
    # Yellow and green sums
    # --------------------------------------------------------------------------------------------------------------------------------------------- #

    # grouped dataframes
    yellow_pickup_grouped = yellow_taxi_df.groupBy("Pickup_Borough").count()
    yellow_dropoff_grouped = yellow_taxi_df.groupBy("Dropoff_Borough").count()
    green_pickup_grouped = green_taxi_df.groupBy("Pickup_Borough").count()
    green_dropoff_grouped = green_taxi_df.groupBy("Dropoff_Borough").count()

    # rename columns
    yellow_pickup_grouped = yellow_pickup_grouped.withColumnRenamed("Pickup_Borough", "Borough").withColumnRenamed("count", "YPickup_Count")
    yellow_dropoff_grouped = yellow_dropoff_grouped.withColumnRenamed("Dropoff_Borough", "Borough").withColumnRenamed("count", "YDropoff_Count")
    green_pickup_grouped = green_pickup_grouped.withColumnRenamed("Pickup_Borough", "Borough").withColumnRenamed("count", "GPickup_Count")
    green_dropoff_grouped = green_dropoff_grouped.withColumnRenamed("Dropoff_Borough", "Borough").withColumnRenamed("count", "GDropoff_Count")
    
    # joins on Borough column
    yellow_sum_df = yellow_pickup_grouped.join(yellow_dropoff_grouped, on="Borough")
    green_sum_df = green_pickup_grouped.join(green_dropoff_grouped, on="Borough")

    # Creation of new sum column
    yellow_sum_df = yellow_sum_df.withColumn("Count_Sum", col("YPickup_Count") + col("YDropoff_Count"))
    green_sum_df = green_sum_df.withColumn("Count_Sum", col("GPickup_Count") + col("GDropoff_Count"))
    
    yellow_sum_df.show()
    green_sum_df.show()

    # order them both (descending order)
    yellow_ordered_df = yellow_sum_df.orderBy(desc("Count_Sum"))
    green_ordered_df = green_sum_df.orderBy(desc("Count_Sum"))

    # get top 5 each
    top_5_yellow = yellow_ordered_df.limit(5)
    top_5_green = green_ordered_df.limit(5)

    top_5_yellow.show()
    top_5_green.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    # Save file
    # --------------------------------------------------------------------------------------------------------------------------------------------- #
    
    # set date/time for file names
    current_time = datetime.now()
    date_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

    # specify that there should only be a single partition
    top_5_yellow = top_5_yellow.coalesce(1)
    top_5_green = top_5_green.coalesce(1)

    # write dataframes out to files in .csv format
    top_5_yellow.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t4_top5_yellow_" + date_time)
    top_5_green.write.mode("overwrite").option("header", True).csv("s3a://" + s3_bucket + "/t4_top5_green_" + date_time)
    
    spark.stop()      