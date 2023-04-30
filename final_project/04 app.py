# Databricks notebook source
# MAGIC %run ./includes/includes

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
from pyspark.sql.functions import monotonically_increasing_id
real_time_inventory_data = real_time_inventory_data.withColumn('index', monotonically_increasing_id())
# removing first n values from diff column and making it null, we will impute our forecast
from pyspark.sql.functions import when, col
diff_new_df = real_time_inventory_data.withColumn('diff_new', when(col('index').between(0, 7), None).otherwise(col('diff')))
display(diff_new_df)

# COMMAND ----------

#loading the model information from Gold table
model_output_df = spark.read.format("delta").load(MODEL_INFO)
display(model_output_df)

model_output = model_output_df.toPandas()
model_output["yhat"] = model_output["yhat"].round()
model_output["capacity"] = 61

staging_forecast_df = model_output[model_output.tag == "Staging"]
prod_forecast_df = model_output[model_output.tag == "Production"]
forecast_df = prod_forecast_df.iloc[-hours_to_forecast:,:]
# creating a new dataframe with just yhat values and index column
forecast_temp = forecast_df[["yhat"]]
forecast_temp["index"] = list(range(0, hours_to_forecast))
forecast_temp = spark.createDataFrame(forecast_temp)

# Join the dataframes based on the index column
merged_df = forecast_temp.join(diff_new_df, on='index', how='outer')

from pyspark.sql.functions import when

#imputing yhat values that we predicted into y
imputed_df = merged_df.withColumn(
    "diff_new", 
    when(merged_df["diff_new"].isNull(), merged_df["yhat"]).otherwise(merged_df["diff_new"])
)

# cumulative addition of  diff_new values to find new_available (our prediction of how many bikes will be available)
from pyspark.sql.functions import col, sum as spark_sum
from pyspark.sql.window import Window
imputed_df = imputed_df.orderBy("hour_window", ascending=True)
window = Window.orderBy("hour_window")
imputed_df = imputed_df.withColumn("new_available", spark_sum(col("diff_new")).over(window))
imputed_df = imputed_df.orderBy("hour_window", ascending=False)

pd_plot = imputed_df.toPandas()
pd_plot = pd_plot.iloc[:hours_to_forecast,:]
pd_plot["capacity"] = STATION_CAPACITY #defined in includes file 

# COMMAND ----------

#plotting the computed forecasts
import plotly.express as px
import plotly.graph_objects as go
fig = go.Figure()
pd_plot["zero_stock"] = 0
fig.add_trace(go.Scatter(x=pd_plot.hour_window, y=pd_plot["new_available"], name='Forecasted available bikes',mode = 'lines+markers',
                         line = dict(color='blue', width=3, dash='solid')))
fig.add_trace(go.Scatter(x=pd_plot.hour_window[:4], y=pd_plot["new_available"][:4], mode = 'markers',name='Forecast for next 4 hours',
                         marker_symbol = 'triangle-up',
                         marker_size = 15,
                         marker_color="green"))
fig.add_trace(go.Scatter(x=pd_plot.hour_window, y=pd_plot["capacity"], name='Station Capacity (Overstock beyond this)',
                         line = dict(color='red', width=3, dash='dot')))
fig.add_trace(go.Scatter(x=pd_plot.hour_window, y=pd_plot["zero_stock"], name='Stock Out (Understock below this)',
                         line = dict(color='red', width=3, dash='dot')))
# Edit the layout
fig.update_layout(title='Forecasted number of available bikes',
                   xaxis_title='Forecasted Timeline',
                   yaxis_title='#bikes',
                   yaxis_range=[-5,100])
fig.show()

# COMMAND ----------

# import gmaps
# gmaps.configure(api_key="AIzaSyCfs22cuSZakjMdM34lNgWYG5Y2eHyiAHQ")
# #40.734814,-73.992085
# new_york_coordinates = (40.730610, -73.975242)
# marker_loc = [(STATION_LAT,STATION_LON)]
# fig = gmaps.figure(center=new_york_coordinates, zoom_level=14)
# markers = gmaps.marker_layer(marker_loc)
# fig.add_layer(markers)
# fig

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
