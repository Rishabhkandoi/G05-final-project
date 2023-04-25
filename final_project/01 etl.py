# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# Reading Live Data

station_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_STATION_INFO_PATH)
    
station_status_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_STATION_STATUS_PATH)
    
weather_df = spark\
    .read\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(BRONZE_NYC_WEATHER_PATH)

# COMMAND ----------

# Reading Historical Data

weather_history = spark\
    .readStream\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .option("ignoreChanges", "true")\
    .format("csv")\
    .load(NYC_WEATHER_FILE_PATH)
    
station_history = spark\
    .readStream\
    .option("inferSchema", "true")\
    .option("header", "true")\
    .option("ignoreChanges", "true")\
    .format("csv")\
    .load(BIKE_TRIP_DATA_PATH)

# COMMAND ----------

# Storing all bronze tables along with appropriate checkpoints

station_df\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_STATION_INFO_DELTA_DIR)\
    .mode("overwrite")\
    .save()


station_status_df\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_STATION_STATUS_DELTA_DIR)\
    .mode("overwrite")\
    .save()

weather_df\
    .write\
    .format("delta")\
    .option("path", REAL_TIME_WEATHER_DELTA_DIR)\
    .mode("overwrite")\
    .save()
    
weather_history\
    .writeStream\
    .format("delta")\
    .option("path", HISTORIC_WEATHER_DELTA_DIR)\
    .option("checkpointLocation", HISTORIC_WEATHER_CHECKPOINT_DIR)\
    .start()
    
station_history\
    .writeStream\
    .format("delta")\
    .option("path", HISTORIC_STATION_INFO_DELTA_DIR)\
    .option("checkpointLocation", HISTORIC_STATION_INFO_CHECKPOINT_DIR)\
    .start()

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
    .readStream\
    .format("delta")\
    .option("ignoreChanges", "true")\
    .load(REAL_TIME_WEATHER_DELTA_DIR)

# weather_history = spark\
#     .readStream\
#     .format("delta")\
#     .option("ignoreChanges", "true")\
#     .load(HISTORIC_WEATHER_DELTA_DIR)
    
# station_history = spark\
#     .readStream\
#     .format("delta")\
#     .option("ignoreChanges", "true")\
#     .load(HISTORIC_STATION_INFO_DELTA_DIR)

weather_history = spark\
    .read\
    .format("delta")\
    .load(HISTORIC_WEATHER_DELTA_DIR)

# station_history = spark\
#     .read\
#     .format("delta")\
#     .load(HISTORIC_STATION_INFO_DELTA_DIR)

# COMMAND ----------

# Join the two dataframes on the 'station_id' column
new_df_station = station_df.join(station_status_df, 'station_id')

# COMMAND ----------



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
    'num_bikes_available',
    'num_docks_available',
    'num_docks_disabled',
    'num_bikes_disabled',
    'last_reported'
)


# COMMAND ----------

new_df_station_filter = new_df_station.filter((col("name") == GROUP_STATION_ASSIGNMENT))

# COMMAND ----------

new_df_station_filter = new_df_station_filter.withColumn("last_reported", col("last_reported").cast("long"))
new_df_station_filter = new_df_station_filter.withColumn("last_reported", date_format(from_unixtime(col("last_reported")), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

trial = new_df_station_filter.select(
    col('last_reported').alias('hour_window'),
    col('name').alias('station_name'),
    col('short_name').alias('station_id'),
    'lat', 
    col('lon').alias('lng'), 
    'num_ebikes_available',
    'num_bikes_available',
    'num_docks_available',
    'num_docks_disabled',
    'num_bikes_disabled',
    'capacity'
    
)
#trial = trial.withColumn("diff", col("in") - col("out"))
trial = trial.withColumn("avail", col("num_ebikes_available")+col("num_bikes_available")+col("num_docks_available")+col("num_docks_disabled")+col("num_bikes_disabled"))

# COMMAND ----------

bike_bronze = trial.select(
    'hour_window',
    'station_name',
    'station_id',
    'lat', 
    'lng', 
    'capacity',
    'avail'
)

# COMMAND ----------

bike_bronze_sorted = bike_bronze.orderBy(col("hour_window"))

# COMMAND ----------

from pyspark.sql.functions import window, last, date_format

df_hourly_availability = (bike_bronze_sorted
  .groupBy(window("hour_window", "1 hour", "1 hour").alias("window_end"))  
  .agg(last("avail").alias("last_availability"))
  .select(date_format("window_end.end", "yyyy-MM-dd HH:mm:ss").alias("hour_window"), "last_availability")
  .orderBy("hour_window"))


# COMMAND ----------

final_stream_bike = df_hourly_availability.select(
    'hour_window',
    col('last_availability').alias('avail'),
)

# COMMAND ----------

display(final_stream_bike)

# COMMAND ----------

#verify kar lena pls

# COMMAND ----------

weather_df = weather_df.withColumn("dt", col("dt").cast("long"))
weather_df = weather_df.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

weather_new = weather_df
# assuming `weather_df` is your DataFrame with the `weather` column
weather_new = weather_new.withColumn("description", col("weather").getItem(0).getField("description"))
weather_new = weather_new.withColumn("icon", col("weather").getItem(0).getField("icon"))
weather_new = weather_new.withColumn("id", col("weather").getItem(0).getField("id"))
weather_new = weather_new.withColumn("main", col("weather").getItem(0).getField("main"))

# drop the `weather` column since we no longer need it
weather_new = weather_new.drop("weather")

# COMMAND ----------

weather_new = weather_new.withColumnRenamed("rain.1h", "rain")

# COMMAND ----------

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

# COMMAND ----------

display(weather_stream)

# COMMAND ----------

# Read NYC_WEATHER
nyc_weather_df = weather_history

# COMMAND ----------

nyc_weather_df = nyc_weather_df.withColumn("dt", col("dt").cast("long"))
nyc_weather_df = nyc_weather_df.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))
# display(nyc_weather_df)

# COMMAND ----------

weather_history = nyc_weather_df.select(
    col("dt").alias("hour_window").cast("string"),
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

# display(weather_history)

# COMMAND ----------

# bike_bronze\
#     .writeStream\
#     .format("delta")\
#     .option("path", REAL_TIME_INVENTORY_INFO_DELTA_DIR)\
#     .option("checkpointLocation", REAL_TIME_INVENTORY_INFO_CHECKPOINT_DIR)\
#     .start()

# COMMAND ----------

# Code to apply transformations in historic data

def apply_transformations(weather_history):
    
    # Read Historic data from bronze storage
    station_history = spark\
        .read\
        .format("delta")\
        .load(HISTORIC_STATION_INFO_DELTA_DIR)

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

    final_bike_historic_weather_merged = final_bike_historic.join(weather_history, on="hour_window", how="left")

    return final_bike_historic_weather_merged

# COMMAND ----------

# dbutils.fs.rm(HISTORIC_INVENTORY_INFO_DELTA_DIR, True)

# COMMAND ----------

# Overwrite the historic transformed data in silver storage if there is any new data according to the timestamp

try:
    latest_end_timestamp_in_silver_storage = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR).select("hour_window").sort(desc("hour_window")).head(1)[0][0]
except:
    latest_end_timestamp_in_silver_storage = '2003-02-28 13:33:07'
latest_start_timestamp_in_bronze = spark.read.format("delta").load(HISTORIC_STATION_INFO_DELTA_DIR).select("started_at").filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT) | (col("end_station_name") == GROUP_STATION_ASSIGNMENT)).sort(desc("ended_at")).head(1)[0][0]

if latest_start_timestamp_in_bronze >= latest_end_timestamp_in_silver_storage:
    print("Overwriting historic data in Silver Storage")
    final_bike_historic_trial = apply_transformations(weather_history)
    final_bike_historic_trial\
        .write\
        .mode("overwrite")\
        .format("delta")\
        .option("path", HISTORIC_INVENTORY_INFO_DELTA_DIR)\
        .save()

# COMMAND ----------

display(spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR))

# COMMAND ----------

relevant_bike_df = spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR)

# COMMAND ----------

relevant_bike_df = relevant_bike_df.withColumn('year',year(relevant_bike_df["hour_window"])).withColumn('month',month(relevant_bike_df["hour_window"])).withColumn('dom',dayofmonth(relevant_bike_df["hour_window"]))
relevant_bike_df = relevant_bike_df.withColumn("year_month",concat_ws("-",relevant_bike_df.year,relevant_bike_df.month))
relevant_bike_df = relevant_bike_df.withColumn("simple_dt",concat_ws("-",relevant_bike_df.year_month,relevant_bike_df.dom))
display(relevant_bike_df)

# COMMAND ----------

import numpy as np

relevant_bike_df_pd = relevant_bike_df.toPandas()
relevant_bike_df_pd_agg = relevant_bike_df_pd.groupby('simple_dt').agg(avg_temp=('feels_like', np.mean),
                                                     avg_uvi=('uvi', np.mean),
                                                     avg_ws=('wind_speed', np.mean),
                                                     avg_humidity=('humidity', np.mean),
                                                     avg_pressure=('pressure', np.mean),
                                                     avg_clouds=('clouds', np.mean),
                                                     avg_visibility=('visibility', np.mean),
                                                     avg_rain_1h=('rain', np.mean),
                                                    #  avg_snow_1h=('snow_1h', np.mean),
                                                     avg_wind_deg=('wind_deg', np.mean),
                                                     avg_dew_point=('dew_point', np.mean))


# COMMAND ----------

relevant_bike_df_pd_agg = relevant_bike_df_pd_agg.reset_index()

# COMMAND ----------

final_df = relevant_bike_df_pd.merge(relevant_bike_df_pd_agg,on="simple_dt",how="left")

# COMMAND ----------

final_df['temp'].isna().replace(final_df.avg_temp,inplace=True)

# COMMAND ----------

pd.plot(final_df.temp)

# COMMAND ----------

display(spark.read.format("delta").load(HISTORIC_INVENTORY_INFO_DELTA_DIR).filter((col("hour_window") > "2022-07-04 01:00:00")))

# COMMAND ----------

display(spark.read.format("delta").load(HISTORIC_STATION_INFO_DELTA_DIR).filter(((col("started_at") > "2022-07-04 01:00:00") & (col("start_station_name") == GROUP_STATION_ASSIGNMENT) & (col("started_at") < "2022-07-04 02:00:00")) | ((col("ended_at") > "2022-07-04 01:00:00") & (col("end_station_name") == GROUP_STATION_ASSIGNMENT) & (col("ended_at") < "2022-07-04 02:00:00"))))


# COMMAND ----------

display(spark.read.format("delta").load(REAL_TIME_STATION_STATUS_DELTA_DIR)\
    .withColumn("last_reported", date_format(from_unixtime(col("last_reported").cast("long")), "yyyy-MM-dd HH:mm:ss"))\
    .filter(col("station_id") == GROUP_STATION_ID)\
        .select("num_bikes_available", "num_ebikes_available", "num_docks_available", "num_scooters_available", "last_reported").sort(asc("last_reported")))

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
