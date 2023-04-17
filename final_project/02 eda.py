# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

import datetime
from pyspark.sql.functions import year, month, dayofmonth,concat_ws,col,sum, max, min, avg, count,from_unixtime, date_format,lit,to_date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql import functions as F



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
df_station_info = spark.readStream.format("delta").load(BRONZE_STATION_INFO_PATH)
df_station_status = spark.readStream.format("delta").load(BRONZE_STATION_STATUS_PATH)
df_weather_data = spark.readStream.format("delta").load(BRONZE_NYC_WEATHER_PATH)
df_bike_trip = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(BIKE_TRIP_DATA_PATH)
df_nyc_weather = spark.read.option("inferSchema", "true").option("header", "true").format("csv").load(NYC_WEATHER_FILE_PATH)
relevant_bike_df = df_bike_trip.filter((df_bike_trip["start_station_name"] == GROUP_STATION_ASSIGNMENT) | (df_bike_trip["end_station_name"] == GROUP_STATION_ASSIGNMENT))

df_nyc_weather = df_nyc_weather.withColumn("dt", col("dt").cast("long"))
df_nyc_weather = df_nyc_weather.withColumn("dt", date_format(from_unixtime(col("dt")), "yyyy-MM-dd HH:mm:ss"))
# First subset for start_station = University
bike_start = df_bike_trip.filter((col("start_station_name") == "University Pl & E 14 St")).select(
    "ride_id", "rideable_type", "started_at", "start_station_name", "start_station_id", "start_lat", "start_lng","member_casual")

# First subset for start_station = University
bike_end = df_bike_trip.filter((col("end_station_name") == "University Pl & E 14 St")).select(
    "ride_id", "rideable_type", "ended_at", "end_station_name", "end_station_id", "end_lat", "end_lng","member_casual")
bike_end = bike_end.withColumnRenamed("ended_at", "started_at").withColumnRenamed("end_station_name", "start_station_name").withColumnRenamed("end_station_id", "start_station_id").withColumnRenamed("end_lat", "start_lat").withColumnRenamed("end_lng", "start_lng")
    
display(bike_end)

relevant_bike_df = bike_end.union(bike_start)
relevant_bike_df = relevant_bike_df.withColumn('year',year(relevant_bike_df["started_at"])).withColumn('month',month(relevant_bike_df["started_at"])).withColumn('dom',dayofmonth(relevant_bike_df["started_at"]))
relevant_bike_df = relevant_bike_df.withColumn("year_month",concat_ws("-",relevant_bike_df.year,relevant_bike_df.month))
relevant_bike_df = relevant_bike_df.withColumn("simple_dt",concat_ws("-",relevant_bike_df.year_month,relevant_bike_df.dom))
relevant_bike_df = relevant_bike_df.withColumn('Hour', hour(relevant_bike_df.started_at))
display(relevant_bike_df)

# COMMAND ----------

# What are the monthly trip trends for your assigned station?
#Create a new column of year-month that aggregates the date as year_month for viz
month_trips= relevant_bike_df.groupBy('year_month').agg(count('ride_id'))
#month_trips = month_trips.withColumn("sort_col",concat_ws("-",month_trips.year_month,lit(1)))
month_df = month_trips.toPandas()

fig = px.bar(month_df, x='year_month', y='count(ride_id)')
fig.show()

# COMMAND ----------

# What are the monthly trip trends for your assigned station?
#Create a new column of year-month that aggregates the date as year_month for viz
hour_trips= relevant_bike_df.groupBy('Hour').agg(count('ride_id'))
#month_trips = month_trips.withColumn("sort_col",concat_ws("-",month_trips.year_month,lit(1)))
hour_df = hour_trips.toPandas()

fig = px.bar(hour_df, x='Hour', y='count(ride_id)')
fig.show()

# COMMAND ----------

# What are the daily trip trends for your given station?
daily_trips = relevant_bike_df.groupBy('simple_dt').agg(count('ride_id'))
#month_trips = month_trips.withColumn("sort_col",concat_ws("-",month_trips.year_month,lit(1)))
daily_trips = daily_trips.toPandas()
daily_trips['simple_dt'] = pd.to_datetime(daily_trips['simple_dt'], format='%Y-%m-%d')
daily_trips.sort_values(by='simple_dt',ascending=False,inplace=True)
fig = px.line(daily_trips, x='simple_dt', y='count(ride_id)')
fig.show()

# COMMAND ----------

# How does a holiday affect the daily (non-holiday) system use trend?
#daily_trips["weekday_type"] = daily_trips.simple_dt.dt.dayofweek
daily_trips["is_weekend"] = daily_trips.simple_dt.dt.dayofweek > 4
fig = px.bar(daily_trips, x='simple_dt', y='count(ride_id)',color="is_weekend")
fig.show()

# COMMAND ----------

#classic/electric?
bike_type_df = relevant_bike_df.groupBy('year_month','rideable_type').agg(count('ride_id'))
#month_trips = month_trips.withColumn("sort_col",concat_ws("-",month_trips.year_month,lit(1)))
bike_type_df = bike_type_df.toPandas()
fig = px.bar(bike_type_df, x='year_month', y='count(ride_id)',color='rideable_type')
fig.show()

# COMMAND ----------

#member/casual?
member_type_df = relevant_bike_df.groupBy('year_month','member_casual').agg(count('ride_id'))
#month_trips = month_trips.withColumn("sort_col",concat_ws("-",month_trips.year_month,lit(1)))
member_type_df = member_type_df.toPandas()
fig = px.bar(member_type_df, x='year_month', y='count(ride_id)',color='member_casual')
fig.show()

# COMMAND ----------

df_nyc_weather = df_nyc_weather.withColumn('year',year(df_nyc_weather["dt"])).withColumn('month',month(df_nyc_weather["dt"])).withColumn('dom',dayofmonth(df_nyc_weather["dt"]))
df_nyc_weather = df_nyc_weather.withColumn("year_month",concat_ws("-",df_nyc_weather.year,df_nyc_weather.month))
df_nyc_weather = df_nyc_weather.withColumn("simple_dt",concat_ws("-",df_nyc_weather.year_month,df_nyc_weather.dom))

# COMMAND ----------

df_nyc_weather = df_nyc_weather.withColumn('Hour', hour(df_nyc_weather.dt))

# COMMAND ----------


df_nyc_weather_pd = df_nyc_weather.toPandas()
weather_hour = df_nyc_weather_pd.groupby('Hour').agg(avg_temp=('feels_like', np.mean),
                                                     avg_uvi=('uvi', np.mean),
                                                     avg_ws=('wind_speed', np.mean),
                                                     avg_humidity=('humidity', np.mean),
                                                     avg_pressure=('pressure', np.mean),
                                                     avg_clouds=('clouds', np.mean),
                                                     avg_visibility=('visibility', np.mean),
                                                     avg_rain_1h=('rain_1h', np.mean),
                                                     avg_snow_1h=('snow_1h', np.mean),
                                                     avg_wind_deg=('wind_deg', np.mean),
                                                     avg_dew_point=('dew_point', np.mean))
final_df_hr = hour_df.join(weather_hour,on="Hour")


# COMMAND ----------


df_nyc_weather_pd = df_nyc_weather.toPandas()
df_nyc_weather_pd['simple_dt'] = pd.to_datetime(df_nyc_weather_pd['simple_dt'], format='%Y-%m-%d')
weather = df_nyc_weather_pd.groupby('simple_dt').agg(avg_temp=('feels_like', np.mean),
                                                     avg_uvi=('uvi', np.mean),
                                                     avg_ws=('wind_speed', np.mean),
                                                     avg_humidity=('humidity', np.mean),
                                                     avg_pressure=('pressure', np.mean),
                                                     avg_clouds=('clouds', np.mean),
                                                     avg_visibility=('visibility', np.mean),
                                                     avg_rain_1h=('rain_1h', np.mean),
                                                     avg_snow_1h=('snow_1h', np.mean),
                                                     avg_wind_deg=('wind_deg', np.mean),
                                                     avg_dew_point=('dew_point', np.mean))
final_df = daily_trips.join(weather,on="simple_dt")


# COMMAND ----------

df_corr = final_df.corr()
mask = np.zeros_like(df_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
fig = px.imshow(df_corr_viz, text_auto=True)
fig.show()

# COMMAND ----------



"""fig = go.Figure()
fig = px.line(final_df, x='simple_dt', y='avg_temp')
fig.add_bar(x=final_df.simple_dt, y=final_df['count(ride_id)'])
fig.show()"""

fig = go.Figure()
fig.add_trace(go.Bar(x=final_df.simple_dt, y=final_df['count(ride_id)'],
                     name="Daily trips", yaxis='y1'))
fig.add_trace(go.Line(x=final_df.simple_dt, y=final_df.avg_uvi,name="Temperature", yaxis="y2"))

# Create axis objects
fig.update_layout(
   xaxis=dict(domain=[0.15, 0.15]),

# create first y axis
yaxis=dict(
   title="Daily ride",
   titlefont=dict(color="green"),
   tickfont=dict(color="blue")
),

# Create second y axis
yaxis2=dict(
   title="Avg. Temperature",
   overlaying="y",
   side="right",
   position=1)
)

fig.update_layout(title_text="Daily Trips Vs. Avg. daily Temperature",
width=716, height=400)
fig.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
