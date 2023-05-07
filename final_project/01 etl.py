# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda for this notebook
# MAGIC - Set Partitions.
# MAGIC - Set Bronze, Silver and Gold Directories.
# MAGIC - Create bronze and silver tables, along with properly documented storage and transformations strategies.
# MAGIC - Create end table for merged inventory bike info and merged weather info to be used in modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC # Setting Shuffle Partitions to number of cores

# COMMAND ----------

# Setting Shuffle Partitions to number of cores

spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism)
print(spark.conf.get("spark.sql.shuffle.partitions"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze Data Dictionary

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

# MAGIC %md
# MAGIC ### Streaming Station Info Table
# MAGIC
# MAGIC - <b>Primary use case</b>: To fetch the capacity of a particular station.
# MAGIC - <b>Data updation frequency</b>: Data Source is updated in every 30min, but we implemented it with a simple read, and thus it would get updated in our bronze table every time this notebook is run.
# MAGIC - <b>Data format</b>: Source and destination both are in delta format.
# MAGIC - <b>Benefit</b>: This table is simply replicated at our end without any changes for being fail-safe at any moment in time.
# MAGIC - <b>Partitioning</b>: To make the system scalable, the data data is partitioned at station level so that any query related to a particular station can be looked for in a single delta file, and thus aggregating all individual transformations at the end. Since there is no particular use case to have any interaction among multiple stations, it is best to partition likewise. Also, since our use case is limited to a single station, we have filtered the source dataset to have only 3 stations (including ours), and thus only 3 individual delta files would be created without spending much time.

# COMMAND ----------

station_df_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Station Status Table
# MAGIC - <b>Primary use case</b>: To fetch the current status (availability of bikes) at a particular station.
# MAGIC - <b>Data updation frequency</b>: Data Source is updated in every 30min, but we implemented it with a simple read, and thus it would get updated in our bronze table every time this notebook is run.
# MAGIC - <b>Data format</b>: Source and destination both are in delta format.
# MAGIC - <b>Benefit</b>: This table is simply replicated at our end without any changes for being fail-safe at any moment in time.
# MAGIC - <b>Partitioning</b>: To make the system scalable, the data data is partitioned at station level so that any query related to a particular station can be looked for in a single delta file, and thus aggregating all individual transformations at the end. Since there is no particular use case to have any interaction among multiple stations, it is best to partition likewise. Also, since our use case is limited to a single station, we have filtered the source dataset to have only 3 stations (including ours), and thus only 3 individual delta files would be created without spending much time.

# COMMAND ----------

station_status_df_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Streaming Weather Table
# MAGIC - <b>Primary use case</b>: To fetch the current weather information (temp, visibility, rain, etc) in New York City, at hourly level.
# MAGIC - <b>Data updation frequency</b>: Data Source is updated in every 30min, but we implemented it with a simple read, and thus it would get updated in our bronze table every time this notebook is run.
# MAGIC - <b>Data format</b>: Source and destination both are in delta format.
# MAGIC - <b>Benefit</b>: This table is simply replicated at our end without any changes for being fail-safe at any moment in time.

# COMMAND ----------

weather_df_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Historic Station Status Table
# MAGIC - <b>Primary use case</b>: To fetch the number of bikes availability at a particular station, in a particular time window.
# MAGIC - <b>Data updation frequency</b>: Since it is implemented with readStream, any new file added in the data path will get updated at our end once this notebook is run. The stream is configured to be triggered only once in order to avoid handling data de-duplicacy explicitly, and limit the CPU usage indefinitely. Since the frequency of dropping any new file in the data path would be something like once a month, it was best to apply the constraint of triggering only once.
# MAGIC - <b>Data format</b>: Source is csv format and destination is in delta format.
# MAGIC - <b>Benefit</b>: This table is simply replicated at our end by converting from csv to delta format without any data level changes for being fail-safe at any moment in time.

# COMMAND ----------

station_history_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Historic Weather Info Table
# MAGIC - <b>Primary use case</b>: To fetch the historic weather information (temp, visibility, rain, etc) in New York City, at hourly level.
# MAGIC - <b>Data updation frequency</b>: Since it is implemented with readStream, any new file added in the data path will get updated at our end once this notebook is run. The stream is configured to be triggered only once in order to avoid handling data de-duplicacy explicitly, and limit the CPU usage indefinitely. Since the frequency of dropping any new file in the data path would be something like once a month, it was best to apply the constraint of triggering only once.
# MAGIC - <b>Data format</b>: Source is csv format and destination is in delta format.
# MAGIC - <b>Benefit</b>: This table is simply replicated at our end by converting from csv to delta format without any data level changes for being fail-safe at any moment in time.

# COMMAND ----------

weather_history_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Table Definitions

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

# MAGIC %md
# MAGIC #### Creating a new column for storing date at hourly format, in station history table, to enable optimized querying with the use of Z-Ordering.

# COMMAND ----------

# Will apply Z-Ordering on hour column

station_history = station_history.filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT)).withColumn("hour", when(col("start_station_name") == GROUP_STATION_ASSIGNMENT, date_format(col("started_at"), "yyyy-MM-dd HH")).otherwise(date_format(col("ended_at"), "yyyy-MM-dd HH")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Info Transformation from Bronze to Silver
# MAGIC
# MAGIC #### Columns defined
# MAGIC - <b>Timestamp (hour_window)</b>: Converted Unix Timestamp to readable time format (data_type=string).
# MAGIC - <b>Is Weekend (is_weekend)</b>: By extracting day of the week from the date column defined above, Saturday and Sunday are classified as 1, while remaining as 0 (data_type=integer).
# MAGIC - <b>Temperature (feels_like)</b>: Same as already present in the table (data_type=double).
# MAGIC - <b>Clouds (clouds)</b>: Same as already present in the table (data_type=long).
# MAGIC
# MAGIC #### Reasoning
# MAGIC - feels_like, clouds and is_weekend show high correlation than other features
# MAGIC - feels_like over temp: feels_like is better indicator of how it feels on the body
# MAGIC - is_weekend: provides insight about trends of weekday vs weekends
# MAGIC
# MAGIC #### Steps
# MAGIC - Transforming historic and stream weather tables with respect to the column definitions defined above.
# MAGIC - Fetching the latest date-time from historic transformed table.
# MAGIC - Filtering the streaming transformed weather table for date greater than the timestamp defined in the previous step.
# MAGIC - Merging both the transformed tables thus formed.

# COMMAND ----------

# Transforming Real Time Weather info

weather_stream = weather_df.withColumn("dt", date_format(from_unixtime(col("dt").cast("long")), "yyyy-MM-dd HH:mm:ss")).withColumn("is_weekend", (dayofweek(col("dt")) == 1) | (dayofweek(col("dt")) == 7)).withColumn("is_weekend", col("is_weekend").cast("int"))
weather_stream = weather_stream.withColumnRenamed("rain.1h", "rain")
weather_stream = weather_stream.select(
    col("dt").alias("hour_window").cast("string"),
    col("feels_like"),
    col("clouds"),
    col("is_weekend")
)

# COMMAND ----------

# Transforming Historical Weather info

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

weather_merged.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Streaming Bike Info Transformation from Bronze to Silver
# MAGIC
# MAGIC #### Columns defined
# MAGIC - <b>Timestamp (hour_window)</b>: Converted Unix Timestamp to readable time format (data_type=string). It is the end-time of the hourly window.
# MAGIC - <b>Remaining Capacity (diff)</b>: It is defined by number of bikes available and number of bikes disabled for the latest time slot in the hourly window, and it's difference with the total capacity of the station (data_type=long).
# MAGIC
# MAGIC #### Reasoning
# MAGIC - Hour window: Since the weather info is at hourly level, with weather features strongly affecting ride count as will be seen in EDA, the bike information is also aggregated at an hourly level.
# MAGIC - Diff: Number of bikes available and disabled when added across all timestamps showed a good distribution to be around the total capacity of the station, and thus was used to calculate the total number of bikes available at the station. This value was then subtracted from the total capactity of the station to provide the remaining capacity of the station.
# MAGIC
# MAGIC #### Steps
# MAGIC - Transforming the date column.
# MAGIC - Aggregating at the hourly window at extracting the last timestamp from each individual aggregation group of hourly window.
# MAGIC - Applying maths as defined above to calculate the remaining capacity of the station for that hourly window.

# COMMAND ----------

# Transforming and saving Real-time Bike Information

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

final_stream_bike = final_stream_bike.withColumn("diff", 61 - col("avail")).select("hour_window", "diff")

final_stream_bike\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_INVENTORY_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()

# COMMAND ----------

final_stream_bike.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Historic Bike Info Transformation from Bronze to Silver
# MAGIC
# MAGIC #### Columns defined
# MAGIC - <b>Timestamp (hour_window)</b>: Converted Unix Timestamp to readable time format (data_type=string). It is the end-time of the hourly window.
# MAGIC - <b>Inventory Change (diff)</b>: For the hourly window, it is defined by total number of incoming bikes (end_station as ours) minus total number of the outgoing bikes (start_station as ours) (data_type=long).
# MAGIC
# MAGIC #### Reasoning
# MAGIC - Hour window: Since the weather info is at hourly level, with weather features strongly affecting ride count as will be seen in EDA, the bike information is also aggregated at an hourly level.
# MAGIC - Diff: Since there was no information on the initial bikes available, it was not possible to calculate the absolute number of available bikes at any time frame. So, the difference between incoming and outgoing bikes was calculated, which is nothing but inventory change in any hour window, irrespective of the initial number of available bikes.
# MAGIC
# MAGIC #### Steps
# MAGIC - Transforming the date column.
# MAGIC - Forming the hourly window by splitting the data with pairs of (start-time, start-station), (end-time, end-station). Since any row having the start station as ours, the end station information is not very useful as of now, and similarly the case with the other split, and thus it helped performing aggregations separately for creating the hour window on each split.
# MAGIC - Perform aggregation on both the splits individually based upon the hour window formation, and thus counting the ride count in the aggregation process. For the (start-time, start-station) split, number of ride counts will be evaluated from "out" column while "in" column will be used in the other split.
# MAGIC - Since there are various hour timestamp for which data might be missing, a dummy dataframe was created for each hour right from the earliest start-time to the latest end-time, each with 0 ride count. This would help in maintaining value for each hour.
# MAGIC - All three dataframes thus formed will be merged, with the logic of "in" (2nd split) - "out" (first split) + 0 (dummy). This will give the value of inventory change for each hourly window.
# MAGIC
# MAGIC #### Additional Logic for performance improvement
# MAGIC - Since history data is likely to be updated once in a month or two to keep it upto date, there would be some data with some additional timestamps each time this is done.
# MAGIC - Thus, simply tracking if there is any additional timestamp detected in the bronze data, when compared to the data already stored in the Silver Table (by comparing latest timestamp from the bronze storage, with the latest timestamp from the silver storage), it would indicate the need to perform the write operation, otherwise just ignore the transformations.
# MAGIC - This would allow all transformations to happen only when any new data arrives, which is a rare event, and thus a whole piece of transformations can be avoided in usual scenario.
# MAGIC
# MAGIC #### Z-Order Implementation
# MAGIC - Since the most critical and heavy query here would be to perform aggregation on the hourly window, so the new feature created earlier ("Hour Timestamp") is used for Z-order optimization so that the aggregations inside the hour window would be quick.

# COMMAND ----------

# Code to apply transformations on historic data

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
    final_bike_historic_trial = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR)
    latest_end_timestamp_in_silver_storage = final_bike_historic_trial.select("hour_window").sort(desc("hour_window")).head(1)[0][0]
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

final_bike_historic_trial.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Inventory Info (Silver Storage) - Merging Historic and Streaming Bike Info
# MAGIC
# MAGIC #### Columns defined
# MAGIC - <b>Timestamp (hour_window)</b>: Converted Unix Timestamp to readable time format (data_type=string). It is the end-time of the hourly window.
# MAGIC - <b>Inventory Change (diff)</b>: For the hourly window, it is defined as the change in inventory (data_type=long).
# MAGIC
# MAGIC #### Reasoning (Primary Use Case)
# MAGIC - It can be used directly for modeling purpose.
# MAGIC
# MAGIC #### Steps
# MAGIC - Fetching the latest hour_window from historic bike info silver storage.
# MAGIC - Calculating inventory change for the streaming data (by lagged difference in the remaining capacities at each hour window), to format it in the same way as the historic data before merging them.
# MAGIC - Filtering the streaming transformed weather table for date greater than the timestamp defined in the previous step.
# MAGIC - Merging both the transformed tables thus formed.

# COMMAND ----------

# Merge Historic and Real Time Bike Inventory Info

historic_inventory_data = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR)
real_time_inventory_data = spark.read.format("delta").load(REAL_TIME_INVENTORY_INFO_DELTA_DIR)

latest_end_timestamp_in_silver_storage = historic_inventory_data.select("hour_window").sort(desc("hour_window")).head(1)[0][0]
latest_end_timestamp_in_silver_storage

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, col
windowSpec = Window.partitionBy().orderBy("hour_window")
df_with_lag = real_time_inventory_data.withColumn("lag_diff", lag("diff").over(windowSpec))
df_with_new_diff = df_with_lag.withColumn("new_diff", col("diff") - col("lag_diff"))
df_filtered = df_with_new_diff.filter(col("hour_window") > latest_end_timestamp_in_silver_storage)
df_final = df_filtered.drop('diff', 'lag_diff')
real_time_inventory_data = df_final.withColumnRenamed('new_diff', 'diff')

# COMMAND ----------

merged_inventory_data = historic_inventory_data.union(real_time_inventory_data)
merged_inventory_data = merged_inventory_data.orderBy("hour_window", ascending=False)
merged_inventory_data\
    .write\
    .format("delta")\
    .option("path", INVENTORY_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()


# COMMAND ----------

# MAGIC %md
# MAGIC # Gold Table (Model Info)
# MAGIC
# MAGIC #### Columns
# MAGIC - <b>Timestamp (ds)</b>: Hourly Timestamps (date_type=timestamp).
# MAGIC - <b>Original Inventory Change (y)</b>: Original value of diff from INVENTORY_INFO table created above (data_type=double).
# MAGIC - <b>Forecasted Inventory Change (yhat)</b> Forecasted values of inventory change (data_type=double).
# MAGIC - <b>Residual (residual)</b> Difference between y and yhat (data_type=double).
# MAGIC - <b>Tag (tag)</b> Staging / Production / Archival (data_type=string).
# MAGIC - <b>Metric (mae)</b> Mean Absolute Error value (data_type=double).
# MAGIC
# MAGIC #### Reasoning
# MAGIC - Promote Model Use Case: Used in the final application to show comparison between staging and production residuals. This helps in deciding whether to promote the staging model to production or not.
# MAGIC - Capacity Utilization: Used in the final application to forecast whether the total number of bikes at a station is over-stock or under-stock in the scheduled hours to forecast. This would help to maintain the stock appropriately ahead of time.
# MAGIC - Metric usage: Mean absolute error would be used from the table with respect to the staging tag in order to transition any current mlflow experiment to staging by comparing the mae value of the current mlflow experiment and the value from the gold table.
# MAGIC
# MAGIC #### Steps
# MAGIC - Created in mdl file.
# MAGIC - Once any mlflow experiment gets completed, if based upon user input or metric value (mae), the tag of the experiment is Staging or Production, the gold table needs to be updated with appropriate values.
# MAGIC - If the tag is Staging, then simply extract the production related data from the gold table, and merge it with the new forecasted values for the staging, and overwrite the gold table thus formed. 
# MAGIC - If the tag is Production, then simply extract the staging related data from the gold table, and merge it with the new forecasted values for the production, and overwrite the gold table thus formed. 

# COMMAND ----------

spark.read.format("delta").load(MODEL_INFO).printSchema()

# COMMAND ----------

import json

# Return Success#
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
