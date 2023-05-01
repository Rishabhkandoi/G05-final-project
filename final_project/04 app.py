# Databricks notebook source
# MAGIC %run ./includes/includes

# COMMAND ----------

# MAGIC %md
# MAGIC # Agenda for this notebook
# MAGIC - To understand about the assigned station like it's location, capacity, etc.
# MAGIC - To report the model performance(residuals) and forecasts.

# COMMAND ----------

#libraries to be imported
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking.client import MlflowClient
import datetime
from pyspark.sql.functions import *

#fetching the number of hours to be shown in graph using widgets
hours_to_forecast = HOURS_TO_FORECAST

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Current timestamp when the notebook is run

# COMMAND ----------

currentdate = pd.Timestamp.now(tz='US/Eastern').round(freq="H")
fmt = '%Y-%m-%d %H:%M:%S'
currenthour = currentdate.strftime("%Y-%m-%d %H") 
currentdate = currentdate.strftime(fmt) 
print("The current timestamp is:",currentdate)

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Production and Staging Model version

# COMMAND ----------

client = MlflowClient()
prod_model = client.get_latest_versions(GROUP_MODEL_NAME, stages=[PROD])
stage_model = client.get_latest_versions(GROUP_MODEL_NAME, stages=[STAGING])

# COMMAND ----------

print("Details of the current Production Model: ")
print(prod_model)

# COMMAND ----------

print("Details of the current Staging Model: ")
print(stage_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Station Name and Location

# COMMAND ----------

print("Assigned Station: ",GROUP_STATION_ASSIGNMENT)

# COMMAND ----------

#locating the assigned station on google maps
lat = STATION_LAT #defined in includes file 
lon = STATION_LON #defined in includes file 
maps_url = f"https://www.google.com/maps/embed/v1/place?key=AIzaSyAzh2Vlgx7LKBUexJ3DEzKoSwFAvJA-_Do&q={lat},{lon}&zoom=15&maptype=satellite"
iframe = f'<iframe width="100%" height="400px" src="{maps_url}"></iframe>'
displayHTML(iframe)

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Current weather (temp and precipitation)

# COMMAND ----------

weather_data = spark.read.format("delta").load(REAL_TIME_WEATHER_DELTA_DIR).withColumnRenamed("rain.1h", "rain").select("time","temp",'humidity',"pressure","rain","wind_speed","clouds").toPandas()
print("Current Weather:")
print(weather_data[weather_data.time==currentdate].reset_index(drop=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Total docks at this station

# COMMAND ----------

print("Station capacity is",STATION_CAPACITY)

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Total bikes available at this station with different bike types and disabled bikes.

# COMMAND ----------

temp_df_data = spark.read.format("delta").load(REAL_TIME_STATION_STATUS_DELTA_DIR).filter(col("station_id") == GROUP_STATION_ID).withColumn("last_reported", date_format(from_unixtime(col("last_reported").cast("long")), "yyyy-MM-dd HH:mm:ss")).sort(desc("last_reported")).select("last_reported","num_bikes_disabled","num_bikes_available","num_ebikes_available","num_scooters_available")
display(temp_df_data.filter(col("last_reported") <= currenthour).sort(desc("last_reported")).head(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Forecast the available bikes for the next 4 hours and highlight any stock out or full station conditions over the predicted period

# COMMAND ----------

# for next HOURS_TO_FORECAST 
model_info = spark.read.format("delta").load(MODEL_INFO)
model_info = model_info.orderBy(desc("ds"))
filtered_model_info = model_info.filter(model_info.tag == "Production")
new_model_info = filtered_model_info.limit(HOURS_TO_FORECAST)
model_info = new_model_info.drop("y", "tag", "residual", "mae")
from pyspark.sql.functions import to_timestamp, date_format
# converting timestamp
model_info = model_info.withColumn("ds", to_timestamp("ds", "yyyy-MM-dd'T'HH:mm:ss.SSSZ")) \
                       .withColumn("ds", date_format("ds", "yyyy-MM-dd HH:mm:ss"))
model_info = model_info.withColumn("yhat", round("yhat", 0))
from pyspark.sql.functions import lit
model_info = model_info.withColumn("avail", lit(None))
# saving only ds, yhat and avail
from pyspark.sql.functions import asc

last_date = model_info.select("ds").orderBy(asc("ds")).limit(1).collect()[0]["ds"]

#reading bike stream data with hour_window and availability columns
real_time_inventory_data = spark.read.format("delta").load(REAL_TIME_INVENTORY_INFO_DELTA_DIR)
real_time_inventory_data = real_time_inventory_data.orderBy("hour_window", ascending=False)
from pyspark.sql.functions import col
real_time_inventory_data = real_time_inventory_data.withColumnRenamed("diff", "avail")

# calculating diff for every hour_window using lag function
# diff is the difference between bike availability between 2 consecutive hours
from pyspark.sql.functions import col, lag, coalesce
from pyspark.sql.window import Window
w = Window.orderBy("hour_window")
real_time_inventory_data = real_time_inventory_data.withColumn("diff", col("avail") - lag(col("avail"), 1).over(w))
real_time_inventory_data = real_time_inventory_data.withColumn("diff", coalesce(col("diff"), col("avail")))
real_time_inventory_data = real_time_inventory_data.orderBy("hour_window", ascending=False)


# real time streaming data till currentdate (variable which has current time)
trial_new_df = real_time_inventory_data
trial_new_df = trial_new_df.filter(trial_new_df.hour_window < last_date)
new_trial = trial_new_df.select("hour_window", "avail", "diff").orderBy(desc("hour_window"))
#renaming columns for same schema
new_trial = new_trial.withColumnRenamed("diff", "yhat")
new_trial = new_trial.withColumnRenamed("hour_window", "ds")

# reorder columns in new_trial
new_trial = new_trial.select("ds", "avail", "yhat")
# reorder columns in model_info
model_info = model_info.select("ds", "avail", "yhat")
# merging datas
merged_df = model_info.union(new_trial)
merged_df = merged_df.orderBy(desc("ds"))


from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.window import Window
# calculating new_available using window
merged_df = merged_df.orderBy("ds", ascending=True)
window = Window.orderBy("ds")
merged_df = merged_df.withColumn("new_available", spark_sum(col("yhat")).over(window))
merged_df = merged_df.orderBy("ds", ascending=False)
display(merged_df)

# COMMAND ----------

#plotting the computed forecasts
import plotly.express as px
import plotly.graph_objects as go

pd_plot = merged_df.toPandas()
pd_plot = pd_plot.iloc[:hours_to_forecast,:]
pd_plot["capacity"] = STATION_CAPACITY #defined in includes file 

fig = go.Figure()
pd_plot["zero_stock"] = 0
fig.add_trace(go.Scatter(x=pd_plot.ds, y=pd_plot["new_available"], name='Forecasted available bikes',mode = 'lines+markers',
                         line = dict(color='blue', width=3, dash='solid')))
fig.add_trace(go.Scatter(x=pd_plot.ds[-4:], y=pd_plot["new_available"][-4:], mode = 'markers',name='Forecast for next 4 hours',
                         marker_symbol = 'triangle-up',
                         marker_size = 15,
                         marker_color="green"))
fig.add_trace(go.Scatter(x=pd_plot.ds, y=pd_plot["capacity"], name='Station Capacity (Overstock beyond this)',
                         line = dict(color='red', width=3, dash='dot')))
fig.add_trace(go.Scatter(x=pd_plot.ds, y=pd_plot["zero_stock"], name='Stock Out (Understock below this)',
                         line = dict(color='orange', width=3, dash='dot')))
# Edit the layout
fig.update_layout(title='Forecasted number of available bikes',
                   xaxis_title='Forecasted Timeline',
                   yaxis_title='#bikes',
                   yaxis_range=[-5,100])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ■ Monitor the performance of your staging and production models using an appropriate residual plot that illustrates the error in your forecasts.

# COMMAND ----------

# Plot the residuals

fig = px.scatter(
    model_output, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols',
    color='tag'
)
fig.show()

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
