# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda for this notebook
# MAGIC - To understand the underlying correlation between the trip and weather data.  
# MAGIC - To uncover the seasonality pattern of trip data.  
# MAGIC - Insights gathered from this workbook are applied while modelling and hyper-parameter tuning.

# COMMAND ----------

#importing packages that are to used for EDA
import datetime
from pyspark.sql.functions import year, month, dayofmonth,concat_ws,col,sum, max, min, avg, count,from_unixtime, date_format,lit,to_date
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from plotly.subplots import make_subplots


# COMMAND ----------

# MAGIC %md
# MAGIC #### Reading both the historical and bronze files for EDA

# COMMAND ----------

df_station_info = spark.readStream.format("delta").load(REAL_TIME_STATION_INFO_DELTA_DIR)
df_station_status = spark.readStream.format("delta").load(REAL_TIME_STATION_STATUS_DELTA_DIR)
df_weather_data = spark.readStream.format("delta").load(REAL_TIME_WEATHER_DELTA_DIR)
df_bike_trip = spark.read.format("delta").load(HISTORIC_STATION_INFO_DELTA_DIR)
df_nyc_weather = spark.read.format("delta").load(HISTORIC_WEATHER_DELTA_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filtering the entire historical dataset for our assigned station. Creating two different subsets with assigned stations as start station and end station. Merging the two subsets for EDA.

# COMMAND ----------

#subset for start_station
bike_start = df_bike_trip.filter((col("start_station_name") == GROUP_STATION_ASSIGNMENT)).select(
    "ride_id", "rideable_type", "started_at", "start_station_name", "start_station_id", "start_lat", "start_lng","member_casual")

#subset for end_station
bike_end = df_bike_trip.filter((col("end_station_name") == GROUP_STATION_ASSIGNMENT)).select(
    "ride_id", "rideable_type", "ended_at", "end_station_name", "end_station_id", "end_lat", "end_lng","member_casual")
bike_end = bike_end.withColumnRenamed("ended_at", "started_at").withColumnRenamed("end_station_name", "start_station_name").withColumnRenamed("end_station_id", "start_station_id").withColumnRenamed("end_lat", "start_lat").withColumnRenamed("end_lng", "start_lng")

#merging both subsets to get final dataset to be used for EDA
relevant_bike_df = bike_end.union(bike_start)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Adding some transformations like date conversion and formatting.

# COMMAND ----------



#Create new columns for hour, year, year-month, daily dates for bike data
relevant_bike_df = relevant_bike_df.withColumn('year',year(relevant_bike_df["started_at"])).withColumn('month',month(relevant_bike_df["started_at"])).withColumn('dom',dayofmonth(relevant_bike_df["started_at"]))
relevant_bike_df = relevant_bike_df.withColumn("year_month",concat_ws("-",relevant_bike_df.year,relevant_bike_df.month))
relevant_bike_df = relevant_bike_df.withColumn("simple_dt",concat_ws("-",relevant_bike_df.year_month,relevant_bike_df.dom))
relevant_bike_df = relevant_bike_df.withColumn('Hour', hour(relevant_bike_df.started_at))
display(relevant_bike_df)

#Create new columns for hour, year, year-month, daily dates for weather data
df_nyc_weather = df_nyc_weather.withColumn("dt", date_format(from_unixtime(col("dt").cast("long")), "yyyy-MM-dd HH:mm:ss"))
df_nyc_weather = df_nyc_weather.withColumn('year',year(df_nyc_weather["dt"])).withColumn('month',month(df_nyc_weather["dt"])).withColumn('dom',dayofmonth(df_nyc_weather["dt"]))
df_nyc_weather = df_nyc_weather.withColumn("year_month",concat_ws("-",df_nyc_weather.year,df_nyc_weather.month))
df_nyc_weather = df_nyc_weather.withColumn("simple_dt",concat_ws("-",df_nyc_weather.year_month,df_nyc_weather.dom))
df_nyc_weather = df_nyc_weather.withColumn('Hour', hour(df_nyc_weather.dt))


# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examining the monthly trip trends.  
# MAGIC Insights:  
# MAGIC   - Trips are highest in the summer months post March, with the highest in June 2022(around 28K trips)
# MAGIC   - Intuitively, the trips data is lower in the winter months as well.

# COMMAND ----------

month_trips= relevant_bike_df.groupBy('year_month').agg(count('ride_id').alias('#trips'))
month_df = month_trips.toPandas()

fig = px.bar(month_df, x='year_month', y='#trips')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ■  Examining the hourly trip trends.  
# MAGIC Insights:  
# MAGIC   - Trips are highest during the afternoon up until late evening time periods. 
# MAGIC   - It gradually decreases during the early morning and late night hours.
# MAGIC   - The peak times from the dataset also coincide intuitively with the busy office hour timings.

# COMMAND ----------

hour_trips= relevant_bike_df.groupBy('Hour').agg(count('ride_id').alias('#trips'))
hour_df = hour_trips.toPandas()

fig = px.bar(hour_df, x='Hour', y='#trips')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examining the daily trip trends.  
# MAGIC Insights:  
# MAGIC   - No specific pattern  other than the same trend observed in the monthly plot above

# COMMAND ----------

daily_trips = relevant_bike_df.groupBy('simple_dt').agg(count('ride_id').alias('#trips'))
daily_trips = daily_trips.toPandas()
daily_trips['simple_dt'] = pd.to_datetime(daily_trips['simple_dt'], format='%Y-%m-%d')
daily_trips.sort_values(by='simple_dt',ascending=False,inplace=True)


fig = px.line(daily_trips, x='simple_dt', y='#trips')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examine the daily trend with the additional persepctive of the type of weekday, whether it is a weekday or weekend?  
# MAGIC Insights:  
# MAGIC   - The trend on weekdays is higher than weekends usually, however this pattern is not constant and it varies month over month.

# COMMAND ----------


daily_trips["is_weekend"] = daily_trips.simple_dt.dt.dayofweek > 4
fig = px.bar(daily_trips, x='simple_dt', y='#trips',color="is_weekend")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ### ■ Examine the trend of daily trips across the different days of week.
# MAGIC  We are looking for this trends as different charts for each year. Given that we have complete data for 2022, it gives a better perspective of the day-wise trip trends.
# MAGIC Insights:  
# MAGIC   - Over all the three years, Saturday and Sunday are the most unpopular days of the week. Esp. Sunday has the lowest traffic overall.
# MAGIC   - Wednesday is the most popular day in the week.
# MAGIC   - To conclude, we can see that the station is very busy mid-week and comparatively lean on Sat/Sun and even Mon/Tue.

# COMMAND ----------

daily_trips["day_number"] = daily_trips.simple_dt.dt.dayofweek
daily_trips["day_of_week"] = daily_trips.simple_dt.dt.day_name()
daily_trips["year_trip"] = daily_trips.simple_dt.dt.year
fig = make_subplots(rows=3,cols=1)
#subplot for 2021
fig.add_trace(
            go.Bar(x = daily_trips[daily_trips.year_trip == 2021].sort_values(by="day_number")["day_of_week"],
                   y = daily_trips[daily_trips.year_trip == 2021].sort_values(by="day_number")["#trips"],
                   name = "Day-wise trip trend for 2021"),
row=1,
col=1
)

fig.add_trace(
            go.Bar(x = daily_trips[daily_trips.year_trip == 2022].sort_values(by="day_number")["day_of_week"],
                   y = daily_trips[daily_trips.year_trip == 2022].sort_values(by="day_number")["#trips"],
                   name = "Day-wise trip trend for 2022"),
row=2,
col=1
)

fig.add_trace(
            go.Bar(x = daily_trips[daily_trips.year_trip == 2023].sort_values(by="day_number")["day_of_week"],
                   y = daily_trips[daily_trips.year_trip == 2023].sort_values(by="day_number")["#trips"],
                   name = "Day-wise trip trend for 2023"),
row=3,
col=1
)

fig.update_layout(title_text="Daily Trips for each day",
width=1200, height=1600)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examine the monthly trip patterns based on the bike type.  
# MAGIC Insights:
# MAGIC   - There are mainly two types of bikes: Classic and Electric bikes
# MAGIC   - There is an anomaly in data with a third bike type(docked bike) that is seen in the data for the month of June 2022 only.
# MAGIC   - Over the months, the classic bike is more popular than the electric bikes

# COMMAND ----------

bike_type_df = relevant_bike_df.groupBy('year_month','rideable_type').agg(count('ride_id').alias('#trips'))
bike_type_df = bike_type_df.toPandas()

fig = px.bar(bike_type_df, x='year_month', y='#trips',color='rideable_type')
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examine the monthly trip patterns based on the member type.
# MAGIC Insights:
# MAGIC   - There are mainly two types of values : Casual/Members
# MAGIC   - Across all the months, the majority of trips is encountered by members rather than casual trips.

# COMMAND ----------

member_type_df = relevant_bike_df.groupBy('year_month','member_casual').agg(count('ride_id').alias('#trips'))
member_type_df = member_type_df.toPandas()

fig = px.bar(member_type_df, x='year_month', y='#trips',color='member_casual')
fig.show()

# COMMAND ----------


df_nyc_weather_pd = df_nyc_weather.toPandas()
df_nyc_weather_pd['simple_dt'] = pd.to_datetime(df_nyc_weather_pd['simple_dt'], format='%Y-%m-%d')
weather = df_nyc_weather_pd.groupby('simple_dt').agg(avg_feels_like=('feels_like', np.mean),
                                                     avg_uvi=('uvi', np.mean),
                                                     avg_ws=('wind_speed', np.mean),
                                                     avg_humidity=('humidity', np.mean),
                                                     avg_pressure=('pressure', np.mean),
                                                     avg_clouds=('clouds', np.mean),
                                                     avg_visibility=('visibility', np.mean),
                                                     avg_rain_1h=('rain_1h', np.mean),
                                                     avg_snow_1h=('snow_1h', np.mean),
                                                     avg_wind_deg=('wind_deg', np.mean),
                                                     avg_dew_point=('dew_point', np.mean),
                                                     avg_temp=('temp',np.mean))
                                                     
final_df = daily_trips.join(weather,on="simple_dt")


# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examine the correlation between the number of daily trips and weather.  
# MAGIC Insights:
# MAGIC   - There are many pairs in the weather data which have correlation amongst themselves, avg_temp and avg_dew_point,etc
# MAGIC   - Variables like Temp, dewpoint, etc have positive correlation with the #trips.
# MAGIC   - Variables like snow, rain,etc has a strong negative correlation with the #trips.

# COMMAND ----------

df_corr = final_df.corr()
mask = np.zeros_like(df_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
df_corr_viz = df_corr.mask(mask).dropna(how='all').dropna('columns', how='all')
fig = px.imshow(df_corr_viz, text_auto=True)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ■ Examine the trends of daily trips and weather 
# MAGIC Insights:  
# MAGIC   - The pattern is very intuitive, with higher feels-like value, there are higher number of trips.

# COMMAND ----------



fig = go.Figure()
fig.add_trace(go.Bar(x=final_df.simple_dt, y=final_df['#trips'],
                     name="Daily trips", yaxis='y1'))
fig.update_traces(marker_color='blue')
fig.add_trace(go.Line(x=final_df.simple_dt, y=final_df.avg_feels_like,name="Feels-like", yaxis="y2"))

# Create axis objects
fig.update_layout(
   xaxis=dict(domain=[0.15, 0.15]),

# create first y axis
yaxis=dict(
   title="Daily trips",
   titlefont=dict(color="green"),
   tickfont=dict(color="blue")
),

# Create second y axis
yaxis2=dict(
   title="Avg. feels like",
   overlaying="y",
   side="right",
   position=1)
)

fig.update_layout(title_text="Daily Trips Vs. Avg. Feels like",
width=1016, height=600)
fig.show()

# COMMAND ----------



fig = go.Figure()
fig.add_trace(go.Bar(x=final_df.simple_dt, y=final_df['#trips'],
                     name="Daily trips", yaxis='y1'))
fig.update_traces(marker_color='green')
fig.add_trace(go.Line(x=final_df.simple_dt, y=final_df.avg_clouds,name="Clouds", yaxis="y2",line_color='#FFFF8F'))

# Create axis objects
fig.update_layout(
   xaxis=dict(domain=[0.15, 0.15]),

# create first y axis
yaxis=dict(
   title="Daily trips",
   titlefont=dict(color="green"),
   tickfont=dict(color="blue")
),

# Create second y axis
yaxis2=dict(
   title="Avg. Clouds",
   overlaying="y",
   side="right",
   position=1)
)

fig.update_layout(title_text="Daily Trips Vs. Avg. daily Clouds",
width=1016, height=600)
fig.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
