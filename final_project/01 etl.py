# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# Reading Live Data

station_df_data = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_STATION_INFO_PATH)
    
station_status_df_data = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_STATION_STATUS_PATH)
    
weather_df_data = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_NYC_WEATHER_PATH)

# COMMAND ----------

# Reading Historical Data

weather_history_data = spark\
    .readStream\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .option("ignoreChanges", "true")\
    .format("csv")\
    .load(NYC_WEATHER_FILE_PATH)\
    .withColumn("visibility", col("visibility").cast("double"))
    
station_history_data = spark\
    .readStream\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .option("ignoreChanges", "true")\
    .format("csv")\
    .load(BIKE_TRIP_DATA_PATH)

# COMMAND ----------

# Adding only 3 stations so that we can leverage the use of partitioning

station_df_data = station_df_data.filter((col("station_id") == GROUP_STATION_ID) | (col("station_id") == "66de63cd-0aca-11e7-82f6-3863bb44ef7c") | (col("station_id") == "b35ba3c0-d3e8-4b1a-b63b-73a7bb518c9e"))
station_status_df_data = station_status_df_data.filter((col("station_id") == GROUP_STATION_ID) | (col("station_id") == "66de63cd-0aca-11e7-82f6-3863bb44ef7c") | (col("station_id") == "b35ba3c0-d3e8-4b1a-b63b-73a7bb518c9e"))

# COMMAND ----------

# Storing all bronze tables along with appropriate checkpoints

station_df_data\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_STATION_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .partitionBy("station_id")\
    .save()

station_status_df_data\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_STATION_STATUS_DELTA_DIR)\
    .mode("overwrite")\
    .option("overwriteSchema", "true")\
    .partitionBy("station_id")\
    .save()

weather_df_data\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_WEATHER_DELTA_DIR)\
    .mode("overwrite")\
    .save()
    
weather_history_data\
    .writeStream\
    .format("delta")\
    .option("path", HISTORIC_WEATHER_DELTA_DIR)\
    .trigger(once = True)\
    .option("checkpointLocation", HISTORIC_WEATHER_CHECKPOINT_DIR)\
    .start().awaitTermination()
    
station_history_data\
    .writeStream\
    .format("delta")\
    .option("path", HISTORIC_STATION_INFO_DELTA_DIR)\
    .trigger(once = True)\
    .option("checkpointLocation", HISTORIC_STATION_INFO_CHECKPOINT_DIR)\
    .start().awaitTermination()

# COMMAND ----------

# Read all bronze tables

station_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(REAL_TIME_STATION_INFO_DELTA_DIR)
    
station_status_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(REAL_TIME_STATION_STATUS_DELTA_DIR)
    
weather_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(REAL_TIME_WEATHER_DELTA_DIR)

weather_history = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(HISTORIC_WEATHER_DELTA_DIR)

station_history = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(HISTORIC_STATION_INFO_DELTA_DIR)

# COMMAND ----------

# Setting Shuffle Partitions to number of cores

spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism)
print(spark.conf.get("spark.sql.shuffle.partitions"))

# COMMAND ----------

# WIll apply Z-Ordering on hour column

station_history = station_history.filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT)).withColumn("hour", when(col("start_station_name") == GROUP_STATION_ASSIGNMENT, date_format(col("started_at"), "yyyy-MM-dd HH")).otherwise(date_format(col("ended_at"), "yyyy-MM-dd HH")))

# COMMAND ----------

display(weather_df)

# COMMAND ----------

# Transorming Real Time Weather info

weather_stream = weather_df.withColumn("dt", date_format(from_unixtime(col("dt").cast("long")), "yyyy-MM-dd HH:mm:ss")).withColumn("is_weekend", (dayofweek(col("dt")) == 1) | (dayofweek(col("dt")) == 7)).withColumn("is_weekend", col("is_weekend").cast("int"))
weather_stream = weather_stream.withColumnRenamed("rain.1h", "rain")
weather_stream = weather_stream.select(
    col("dt").alias("hour_window").cast("string"),
    col("feels_like"),
    col("clouds"),
    col("is_weekend")
)

# COMMAND ----------

# Transorming Historical Weather info

weather_historical = weather_history.withColumn("dt", date_format(from_unixtime(col("dt").cast("long")), "yyyy-MM-dd HH:mm:ss")).withColumn("is_weekend", (dayofweek(col("dt")) == 1) | (dayofweek(col("dt")) == 7)).withColumn("is_weekend", col("is_weekend").cast("int"))

weather_historical = weather_historical.select(
    col("dt").alias("hour_window").cast("string"),
    col("feels_like"),
    col("clouds"),
    col("is_weekend"))

# COMMAND ----------

# Merging weather data

latest_end_timestamp_for_weather_hist = weather_historical.select("hour_window").sort(desc("hour_window")).head(1)[0][0]
weather_merged = weather_stream.filter(col("hour_window") > latest_end_timestamp_for_weather_hist).union(weather_historical)

weather_merged\
    .write\
    .format("delta")\
    .option("path", WEATHER_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()

# COMMAND ----------

# Transorming and saving Real-time Bike Information

# Join the two dataframes on the 'station_id' column
new_df_station = station_df.join(station_status_df, 'station_id')

# Select the required columns
new_df_station = new_df_station.select(
    'station_id', 
    'name', 
    'region_id', 
    'short_name', 
    'lat', 
    'lon', 
    'capacity',
    'num_bikes_available',
    'num_bikes_disabled',
    'last_reported'
)

new_df_station_filter = new_df_station.filter((col("name") == GROUP_STATION_ASSIGNMENT))

new_df_station_filter = new_df_station_filter.withColumn("last_reported", col("last_reported").cast("long"))
new_df_station_filter = new_df_station_filter.withColumn("last_reported", date_format(from_unixtime(col("last_reported")), "yyyy-MM-dd HH:mm:ss"))

trial = new_df_station_filter.select(
    col('last_reported').alias('hour_window'),
    col('name').alias('station_name'),
    col('short_name').alias('station_id'),
    'lat', 
    col('lon').alias('lng'), 
    'num_bikes_available',
    'num_bikes_disabled',
    'capacity'
    
)

trial = trial.withColumn("avail", col("num_bikes_available")+col("num_bikes_disabled"))

bike_bronze = trial.select(
    'hour_window',
    'station_name',
    'station_id',
    'lat', 
    'lng', 
    'capacity',
    'avail'
)

bike_bronze_sorted = bike_bronze.orderBy(col("hour_window"))

df_hourly_availability = (bike_bronze_sorted
  .groupBy(window("hour_window", "1 hour", "1 hour").alias("window_end"))  
  .agg(last("avail").alias("last_availability"))
  .select(date_format("window_end.end", "yyyy-MM-dd HH:mm:ss").alias("hour_window"), "last_availability")
  .orderBy("hour_window"))

final_stream_bike = df_hourly_availability.select(
    'hour_window',
    col('last_availability').alias('avail'),
)

final_stream_bike = final_stream_bike.withColumn("diff", col("avail") - 61).select("hour_window", "diff")

final_stream_bike\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_INVENTORY_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()

# COMMAND ----------

# Code to apply transformations in historic data

def apply_transformations(weather_history, station_history):

    bike_trip_df = station_history.sort(desc("started_at"))

    # First subset for our start_station
    bike_start = bike_trip_df.filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT)).select(
        "ride_id", "rideable_type", "started_at", "start_station_name", "start_station_id", "start_lat", "start_lng","member_casual")

    # creating window for every hour from the start date and time
    hourly_counts_start = bike_start \
        .groupBy(window(col("started_at"), "1 hour").alias("hour_window")) \
        .agg(count("ride_id").alias("ride_count").alias("out")) \
        .orderBy("hour_window")

    hourly_counts_start = hourly_counts_start.withColumn("hour_window", col("hour_window.end"))

    # Second subset for our end_station
    bike_end = bike_trip_df.filter((col("end_station_name") == GROUP_STATION_ASSIGNMENT)).select(
        "ride_id", "rideable_type", "ended_at", "end_station_name", "end_station_id", "end_lat", "end_lng","member_casual")

    # creating window for every hour from the end date and time

    hourly_counts_end = bike_end \
        .groupBy(window(col("ended_at"), "1 hour").alias("hour_window")) \
        .agg(count("ride_id").alias("ride_count").alias("in")) \
        .orderBy("hour_window")

    hourly_counts_end = hourly_counts_end.withColumn("hour_window", col("hour_window.end"))

    # creating dummy table for every hour and imputing 0 for in and out values 
    # Define start and end dates
    start_date = pd.to_datetime(bike_start.select("started_at").sort(asc("started_at")).head(1)[0][0]).round("H")
    end_date = pd.to_datetime(bike_end.select("ended_at").sort(desc("ended_at")).head(1)[0][0]).round("H")

    # Create a Spark DataFrame with hourly date range and in/out columns initialized to 0
    dummy = spark.range(0, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() // 3600 + 1, step=1)\
        .withColumn("date", lit(pd.to_datetime(start_date)))\
        .withColumn("in", lit(0))\
        .withColumn("out", lit(0))

    # Add 1 hour to each row
    dummy = dummy.rdd.map(lambda x: (x[0], x[1] + pd.Timedelta(hours=x[0]), x[2], x[3])).toDF(['index', 'date', 'in', 'out'])

    #out_dummy table
    out_dummy = dummy.select('date', 'out')
    # rename the 'date' column in out_dummy to 'hour_window' to match the schema of hourly_counts_starts
    out_dummy = out_dummy.withColumnRenamed('date', 'hour_window')

    # left-anti join to fill 0 where no bikes went out for a given hour time frame
    missing_rows_start = out_dummy.join(hourly_counts_start, on='hour_window', how='left_anti')
    hourly_counts_start = hourly_counts_start.union(missing_rows_start.select(hourly_counts_start.columns))

    #re name for in_dummy 
    in_dummy = dummy.select('date','in')
    in_dummy = in_dummy.withColumnRenamed('date', 'hour_window')

    #similarly left-anti join
    missing_rows = in_dummy.join(hourly_counts_end, on='hour_window', how='left_anti')
    hourly_counts_end = hourly_counts_end.union(missing_rows.select(hourly_counts_end.columns))

    #merging both the tables
    merged_table = hourly_counts_start.join(hourly_counts_end, on='hour_window', how='inner')
    final_bike_trip = merged_table.orderBy(col("hour_window"))

    # filling in values for each row
    final_bike_trip = final_bike_trip.withColumn("station_name", lit(GROUP_STATION_ASSIGNMENT)) \
                                    .withColumn("station_id", lit("5905.14")) \
                                    .withColumn("lat", lit("40.734814").cast("double")) \
                                    .withColumn("lng", lit("-73.992085").cast("double"))
                                 
    # converting to yyyy-MM-dd HH:mm:ss format
    final_bike_trip= final_bike_trip.withColumn("hour_window", date_format("hour_window", "yyyy-MM-dd HH:mm:ss"))

    df_bike= final_bike_trip
    df_bike = df_bike.withColumn("diff", col("in") - col("out"))

    # cum sum - diff column
    window_val = (Window.partitionBy('station_name').orderBy('hour_window')
                .rangeBetween(Window.unboundedPreceding, 0))
    cumu_sum_diff = df_bike.withColumn('avail', F.sum('diff').over(window_val))

    #setting initial_bikes 
    initial_bike = 61
    final_bike_historic = cumu_sum_diff.withColumn("avail", cumu_sum_diff["avail"] + lit(initial_bike))

    # final_bike_historic_weather_merged = final_bike_historic.join(weather_history, on="hour_window", how="left")

    final_bike_historic = final_bike_historic.select("hour_window", "diff")

    return final_bike_historic

# COMMAND ----------

# Overwrite the historic transformed data in silver storage if there is any new data according to the timestamp

try:
    latest_end_timestamp_in_silver_storage = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR).select("hour_window").sort(desc("hour_window")).head(1)[0][0]
except:
    latest_end_timestamp_in_silver_storage = '2003-02-28 13:33:07'
latest_start_timestamp_in_bronze = station_history.select("started_at").filter(col("start_station_name") == GROUP_STATION_ASSIGNMENT).sort(desc("started_at")).head(1)[0][0]
latest_end_timestamp_in_bronze = station_history.select("ended_at").filter(col("end_station_name") == GROUP_STATION_ASSIGNMENT).sort(desc("ended_at")).head(1)[0][0]

if latest_start_timestamp_in_bronze >= latest_end_timestamp_in_silver_storage or latest_end_timestamp_in_bronze >= latest_end_timestamp_in_silver_storage:
    print("Overwriting historic data in Silver Storage")
    final_bike_historic_trial = apply_transformations(weather_history, station_history)
    final_bike_historic_trial\
        .write\
        .mode("overwrite")\
        .format("delta")\
        .option("path", HISTORIC_INVENTORY_INFO_DELTA_DIR)\
        .option("zOrderByCol", "hour")\
        .save()

# COMMAND ----------

# Merge Historic and Real Time Bike Inventory Info

historic_inventory_data = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR)
real_time_inventory_data = spark.read.format("delta").load(REAL_TIME_INVENTORY_INFO_DELTA_DIR)

latest_end_timestamp_in_silver_storage = historic_inventory_data.select("hour_window").sort(desc("hour_window")).head(1)[0][0]
real_time_inventory_data = real_time_inventory_data.filter(col("hour_window") > latest_end_timestamp_in_silver_storage)

merged_inventory_data = historic_inventory_data.union(real_time_inventory_data)

merged_inventory_data\
    .write\
    .format("delta")\
    .option("path", INVENTORY_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()


# COMMAND ----------

# dbutils.widgets.text("01.start_date", "2023-04-10", "Start Date")
# dbutils.widgets.text("02.end_date", "2023-05-06", "End Date")
# dbutils.widgets.text("03.hours_to_forecast", "6", "Hours Forecast")
# dbutils.widgets.text("04.promote_model", "yes", "Promote Model")

# COMMAND ----------

# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)
# print("YOUR CODE HERE...")

# COMMAND ----------

import json

# Return Success#
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


