# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

dbutils.widgets.text("01.start_date", "2023-04-10", "Start Date")
dbutils.widgets.text("02.end_date", "2023-05-06", "End Date")
dbutils.widgets.text("03.hours_to_forecast", "6", "Hours Forecast")
dbutils.widgets.text("04.promote_model", "yes", "Promote Model")
start_date = str(dbutils.widgets.get('01.start_date'))
end_date = str(dbutils.widgets.get('02.end_date'))
hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

print(start_date,end_date,hours_to_forecast, promote_model)
print("YOUR CODE HERE...")

# COMMAND ----------

df_station_info = spark.readStream.format("delta").load(BRONZE_STATION_INFO_PATH)
df_station_status = spark.readStream.format("delta").load(BRONZE_STATION_STATUS_PATH)
df_weather_data = spark.readStream.format("delta").load(BRONZE_NYC_WEATHER_PATH)
df_bike_trip = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(BIKE_TRIP_DATA_PATH)
df_nyc_weather = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(NYC_WEATHER_FILE_PATH)

# COMMAND ----------

relevant_bike_df = df_bike_trip.filter((df_bike_trip["start_station_name"] == GROUP_STATION_ASSIGNMENT) | (df_bike_trip["end_station_name"] == GROUP_STATION_ASSIGNMENT))

# COMMAND ----------

display(relevant_bike_df)

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime, date_format
df_nyc_weather = df_nyc_weather.withColumn("dt", col("dt").cast("long"))
df_nyc_weather = df_nyc_weather.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

#To see min and max dates in df_nyc_weather
df_nyc_weather.select("dt").summary().show()

# COMMAND ----------

df_bike_trip.select("started_at").summary().show()
df_bike_trip.select("ended_at").summary().show()

# COMMAND ----------

#This is to conclude that even the weather data has only the relevant latitude and longitudes of our choice.
df_bike_trip.select("start_lat").summary().show()
df_bike_trip.select("start_lng").summary().show()
df_bike_trip.select("end_lat").summary().show()
df_bike_trip.select("end_lng").summary().show()
df_nyc_weather.select("lat").summary().show()
df_nyc_weather.select("lon").summary().show()

# COMMAND ----------

df_nyc_weather = df_nyc_weather.withColumn('date', split(df_nyc_weather['dt'], ' ').getItem(0)).withColumn('timeband_start', split(df_nyc_weather['dt'], ' ').getItem(1))

# COMMAND ----------

display(tumblingWindows)

# COMMAND ----------

from pyspark.sql.functions import *
tumblingWindows = df_bike_trip.withWatermark("started_at", "1 hour")
#tumblingWindows.show(truncate = False)

# COMMAND ----------

display(df_bike_trip)

# COMMAND ----------

from pyspark.sql import functions as F
df3 = df_nyc_weather.join(relevant_bike_df, on=[(df_nyc_weather['dt'] <= relevant_bike_df['started_at'])], how='inner')
df3.select("started_at","ended_at","dt").show()


# COMMAND ----------



# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
