# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

station_df = spark.readStream.format("delta").option("ignoreChanges", "true").load(BRONZE_STATION_INFO_PATH)
station_status_df = spark.readStream.format("delta").option("ignoreChanges", "true").load(BRONZE_STATION_STATUS_PATH)
weather_df = spark.readStream.format("delta").option("ignoreChanges", "true").load(BRONZE_NYC_WEATHER_PATH)

# COMMAND ----------

station_df.printSchema()
display(station_df)

# COMMAND ----------

station_status_df.printSchema()
display(station_status_df)

# COMMAND ----------

from pyspark.sql.functions import col

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
    'num_ebikes_available',
    col('num_bikes_available').alias('bikes_available'), 
    col('num_docks_available').alias('docks_available'),
    'is_renting', 
    'is_returning',
    'last_reported'
)
new_df_station.printSchema()

# COMMAND ----------

new_df_station_filter = new_df_station.filter((col("name") == GROUP_STATION_ASSIGNMENT))
display(new_df_station_filter)

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime, date_format

new_df_station_filter = new_df_station_filter.withColumn("last_reported", col("last_reported").cast("long"))
new_df_station_filter = new_df_station_filter.withColumn("last_reported", date_format(from_unixtime(col("last_reported")), "yyyy-MM-dd HH:mm:ss"))
display(new_df_station_filter)

# COMMAND ----------

trial = new_df_station_filter.select(
    col('last_reported').alias('hour_window'),
    col('is_renting').alias('out'), 
    col('is_returning').alias('in'),
    col('name').alias('station_name'),
    col('short_name').alias('station_id'),
    'lat', 
    col('lon').alias('lng'), 
    col('bikes_available').alias('avail')
)
trial = trial.withColumn("diff", col("in") - col("out"))

# COMMAND ----------

bike_bronze = trial.select(
    'hour_window',
    'out', 
    'in',
    'station_name',
    'station_id',
    'lat', 
    'lng', 
    'diff',
    'avail'
)

display(bike_bronze)

# COMMAND ----------

#from pyspark.sql.functions import window, col, count

#window_size = "1 hour"
#window_col = window(col("last_reported"), window_size, window_size)
#grouped_df = new_df_station_filter.groupBy(window_col).agg(count("*"))

# COMMAND ----------

#grouped_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime, date_format

weather_df = weather_df.withColumn("dt", col("dt").cast("long"))
weather_df = weather_df.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

from pyspark.sql.functions import col
weather_new = weather_df
# assuming `weather_df` is your DataFrame with the `weather` column
weather_new = weather_new.withColumn("description", col("weather").getItem(0).getField("description"))
weather_new = weather_new.withColumn("icon", col("weather").getItem(0).getField("icon"))
weather_new = weather_new.withColumn("id", col("weather").getItem(0).getField("id"))
weather_new = weather_new.withColumn("main", col("weather").getItem(0).getField("main"))

# drop the `weather` column since we no longer need it
weather_new = weather_new.drop("weather")
display(weather_new)

# COMMAND ----------

weather_new = weather_new.withColumnRenamed("rain.1h", "rain")

# COMMAND ----------

from pyspark.sql.functions import col

weather_stream = weather_new.select(
    col("dt").cast("string"),
    col("temp").cast("double"),
    col("feels_like").cast("double"),
    col("pressure").cast("integer"),
    col("humidity").cast("integer"),
    col("dew_point").cast("double"),
    col("uvi").cast("double"),
    col("clouds").cast("integer"),
    col("visibility").cast("integer"),
    col("wind_speed").cast("double"),
    col("wind_deg").cast("integer"),
    col("pop").cast("double"),
    col("id").cast("integer"),
    col("main"),
    col("description"),
    col("icon"),
    col("rain").cast("double")
)

display(weather_stream)

# COMMAND ----------

"""dt:string
temp:double
feels_like:double
pressure:integer
humidity:integer
dew_point:double
uvi:double
clouds:integer
visibility:integer
wind_speed:double
wind_deg:integer
pop:double
id:integer
main:string
description:string
icon:string
rain_1h:double

dt: string, 
temp: double, 
feels_like: double, 
pressure: bigint, 
humidity: bigint, 
dew_point: double, 
uvi: double, 
clouds: bigint, 
visibility: bigint, 
wind_speed: double, 
wind_deg: bigint, 
pop: double, 
id: bigint,
main: string
description: string,
icon: string,
rain.1h: double """

# COMMAND ----------

weather_new

# COMMAND ----------

#Historic Bike Trip Data for Model Building (Stream this data source)

# COMMAND ----------

bike_trip_df = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(BIKE_TRIP_DATA_PATH)

# COMMAND ----------

from pyspark.sql.functions import col, desc
bike_trip_df = bike_trip_df.orderBy(col("started_at").desc())
display(bike_trip_df)

# COMMAND ----------

# Dividing bike_data into two parts , one where start = University and other where end = University
# If it starts at Uni, it means bikes are going out, therefore out count
# If ends at Uni, it means bikes are coming in, therefore in count

# COMMAND ----------

# First subset for start_station = University
bike_start = bike_trip_df.filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT)).select(
    "ride_id", "rideable_type", "started_at", "start_station_name", "start_station_id", "start_lat", "start_lng","member_casual")

sorted_bike_start = bike_start.orderBy(col("started_at").desc())

display(sorted_bike_start)

# COMMAND ----------

# creating window for every hour from the start date and time
from pyspark.sql.functions import window, count

hourly_counts_start = bike_start \
    .groupBy(window(col("started_at"), "1 hour").alias("hour_window")) \
    .agg(count("ride_id").alias("ride_count").alias("out")) \
    .orderBy("hour_window")

hourly_counts_start = hourly_counts_start.withColumn("hour_window", col("hour_window.end"))
display(hourly_counts_start)

# COMMAND ----------

# Second subset for end_station = University
bike_end = bike_trip_df.filter((col("end_station_name") == GROUP_STATION_ASSIGNMENT)).select(
    "ride_id", "rideable_type", "ended_at", "end_station_name", "end_station_id", "end_lat", "end_lng","member_casual")

sorted_bike_end = bike_end.orderBy(col("ended_at"))
display(sorted_bike_end)

# COMMAND ----------

# creating window for every hour from the end date and time

hourly_counts_end = bike_end \
    .groupBy(window(col("ended_at"), "1 hour").alias("hour_window")) \
    .agg(count("ride_id").alias("ride_count").alias("in")) \
    .orderBy("hour_window")


hourly_counts_end = hourly_counts_end.withColumn("hour_window", col("hour_window.end"))

display(hourly_counts_end)

# COMMAND ----------

# creating dummy table for every hour and imputing 0 for in and out values 
from pyspark.sql.functions import lit
import pandas as pd
# Define start and end dates
start_date = '2021-11-01 01:00:00'
end_date = '2023-03-01 00:00:00'

# Create a Spark DataFrame with hourly date range and in/out columns initialized to 0
dummy = spark.range(0, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() // 3600 + 1, step=1)\
    .withColumn("date", lit(pd.to_datetime(start_date)))\
    .withColumn("in", lit(0))\
    .withColumn("out", lit(0))

# Add 1 hour to each row
dummy = dummy.rdd.map(lambda x: (x[0], x[1] + pd.Timedelta(hours=x[0]), x[2], x[3])).toDF(['index', 'date', 'in', 'out'])

# Show the resulting DataFrame
display(dummy)

# COMMAND ----------

#out_dummy table
out_dummy = dummy.select('date', 'out')
# rename the 'date' column in out_dummy to 'hour_window' to match the schema of hourly_counts_starts
out_dummy = out_dummy.withColumnRenamed('date', 'hour_window')
display(out_dummy)

# COMMAND ----------

# left-anti join to fill 0 where no bikes went out for a given hour time frame
from pyspark.sql.functions import col
missing_rows_start = out_dummy.join(hourly_counts_start, on='hour_window', how='left_anti')
hourly_counts_start = hourly_counts_start.union(missing_rows_start.select(hourly_counts_start.columns))
display(hourly_counts_start)

# COMMAND ----------

#re name for in_dummy 
in_dummy = dummy.select('date','in')
in_dummy = in_dummy.withColumnRenamed('date', 'hour_window')
display(in_dummy)

# COMMAND ----------

#similarly left-anti join
from pyspark.sql.functions import col
missing_rows = in_dummy.join(hourly_counts_end, on='hour_window', how='left_anti')
hourly_counts_end = hourly_counts_end.union(missing_rows.select(hourly_counts_end.columns))
display(hourly_counts_end)

# COMMAND ----------

#merging both the tables
merged_table = hourly_counts_start.join(hourly_counts_end, on='hour_window', how='inner')
final_bike_trip = merged_table.orderBy(col("hour_window"))
display(final_bike_trip)

# COMMAND ----------

# filling in values for each row
final_bike_trip = final_bike_trip.withColumn("station_name", lit(GROUP_STATION_ASSIGNMENT)) \
                                 .withColumn("station_id", lit("5905.14")) \
                                 .withColumn("lat", lit("40.734814").cast("double")) \
                                 .withColumn("lng", lit("-73.992085").cast("double"))

display(final_bike_trip)

# COMMAND ----------

from pyspark.sql.functions import date_format
# converting to yyyy-MM-dd HH:mm:ss format
final_bike_trip= final_bike_trip.withColumn("hour_window", date_format("hour_window", "yyyy-MM-dd HH:mm:ss"))
display(final_bike_trip)

# COMMAND ----------

from pyspark.sql.functions import row_number, when
from pyspark.sql.window import Window
df_bike= final_bike_trip

# COMMAND ----------

from pyspark.sql.functions import col, lag, coalesce
# add a column with the difference between in and out
df_bike = df_bike.withColumn("diff", col("in") - col("out"))
display(df_bike)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
# cum sum - diff column
window_val = (Window.partitionBy('station_name').orderBy('hour_window')
             .rangeBetween(Window.unboundedPreceding, 0))
cumu_sum_diff = df_bike.withColumn('avail', F.sum('diff').over(window_val))
display(cumu_sum_diff)

# COMMAND ----------

#setting initial_bikes 
from pyspark.sql.functions import lit
initial_bike = 30
final_bike_historic = cumu_sum_diff.withColumn("avail", cumu_sum_diff["avail"] + lit(initial_bike))
display(final_bike_historic)

# COMMAND ----------

#Historic NYC Weather for Model Building

# COMMAND ----------

# Read NYC_WEATHER
nyc_weather_df = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(NYC_WEATHER_FILE_PATH)

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime, date_format

nyc_weather_df = nyc_weather_df.withColumn("dt", col("dt").cast("long"))
nyc_weather_df = nyc_weather_df.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))
display(nyc_weather_df)

# COMMAND ----------

from pyspark.sql.functions import col

weather_history = nyc_weather_df.select(
    col("dt").cast("string"),
    col("temp").cast("double"),
    col("feels_like").cast("double"),
    col("pressure").cast("integer"),
    col("humidity").cast("integer"),
    col("dew_point").cast("double"),
    col("uvi").cast("double"),
    col("clouds").cast("integer"),
    col("visibility").cast("integer"),
    col("wind_speed").cast("double"),
    col("wind_deg").cast("integer"),
    col("pop").cast("double"),
    col("id").cast("integer"),
    col("main"),
    col("description"),
    col("icon"),
    col("rain_1h").alias("rain").cast("double")
)

display(weather_history)

# COMMAND ----------

bike_bronze_trial = bike_bronze
final_bike_historic_trial = final_bike_historic

# COMMAND ----------

# Create a checkpoint folder
CHECKPOINT_DIR = GROUP_DATA_PATH + "/checkpoints"
dbutils.fs.mkdirs(CHECKPOINT_DIR)

# COMMAND ----------

TABLES_DIR = GROUP_DATA_PATH + "/tables"
dbutils.fs.mkdirs(TABLES_DIR)

# COMMAND ----------

TABLE_NAME = GROUP_DB_NAME + ".bikeinventoryinfo"

(final_bike_historic_trial
 .write
 .format("delta")
 .mode("ignore")
 .saveAsTable(TABLE_NAME)
)

# COMMAND ----------

(bike_bronze_trial.writeStream
 .format("delta")
 .trigger(once = True)
 .option("checkpointLocation", CHECKPOINT_DIR)
 .toTable(TABLE_NAME)
 .start()
 .awaitTermination()
)

# COMMAND ----------

display(spark.read.format("delta").table(TABLE_NAME).sort(-col("hour_window")))

# COMMAND ----------

display(spark.read.format("delta").table(TABLE_NAME))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#merged_df = final_bike_historic_trial.union(bike_bronze_trial)

# COMMAND ----------

#display(merged_df)

# COMMAND ----------

# MERGING BIKE_BRONZE and FINAL_BIKE_HISTORIC

# COMMAND ----------

# MERGING final_bike_historic and and weather data

# COMMAND ----------

#weather_bike_merged = final_bike_historic.join(nyc_weather_df, (final_bike_historic.hour_window == #nyc_weather_df.dt), how="left")
#display(weather_bike_merged)

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

#import json

# Return Success#
#dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


