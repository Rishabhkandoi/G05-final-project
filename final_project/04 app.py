# Databricks notebook source
# MAGIC %pip install geopandas
# MAGIC %pip install contextily
# MAGIC %pip install gmaps

# COMMAND ----------

# MAGIC %run ./includes/includes

# COMMAND ----------

# MLFlow Tracking
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC Each run of the notebook will update/display the following:  
# MAGIC ■ Current timestamp when the notebook is run (now)  
# MAGIC ■ Production Model version  
# MAGIC ■ Staging Model version  
# MAGIC ■ Station name and a map location (marker)  
# MAGIC ■ Current weather (temp and precip)  
# MAGIC ■ Total docks at this station  
# MAGIC ■ Total bikes available at this station  
# MAGIC ■ Forecast the available bikes for the next 4 hours.  
# MAGIC ■ Highlight any stock out or full station conditions over the predicted period.  
# MAGIC ■ Monitor the performance of your staging and production models using an appropriate residual plot that illustrates the error in your forecasts.

# COMMAND ----------

import plotly.express as px
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx


# COMMAND ----------

g05_station = pd.DataFrame(columns=["name","lon","lat"])
g05_station = g05_station.append({'name': GROUP_STATION_ASSIGNMENT, 'lon': STATION_LON, 'lat': STATION_LAT}, ignore_index=True)
print(g05_station)

# COMMAND ----------

g05_station = gpd.GeoDataFrame(g05_station, geometry=gpd.points_from_xy(g05_station.lon, g05_station.lat))
g05_station

# COMMAND ----------

df = gpd.read_file(gpd.datasets.get_path('nybb'))
ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

# COMMAND ----------

df_wm = df.to_crs(epsg=4326)

# COMMAND ----------

df_wm

# COMMAND ----------

df_wm = df_wm[df_wm.BoroCode == 1]
#df_wm = df_wm[df_wm.BoroCode != 1]
#df_wm = df_wm[df_wm.BoroCode != 4]
#df_wm = df_wm[df_wm.BoroCode != 3]

# COMMAND ----------

ax = df_wm.plot(figsize=(10, 10), alpha=0.5)
cx.add_basemap(ax, crs=df_wm.crs)
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, zoom=12   
)
g05_station.plot(markersize=100,ax=ax,color="red")


# COMMAND ----------

API_KEY = "AIzaSyBgtjooD-Bjm96Q-vbHQv72KMYQN8U7ZJU"


# COMMAND ----------

!jupyter nbextension enable --py --sys-prefix widgetsnbextension

# COMMAND ----------

import gmaps
gmaps.configure(api_key="AIzaSyCfs22cuSZakjMdM34lNgWYG5Y2eHyiAHQ")

# COMMAND ----------

STATION_LON

# COMMAND ----------


new_york_coordinates = (STATION_LAT, STATION_LON)
marker_loc = [(STATION_LAT, STATION_LON)]
fig = gmaps.figure(center=new_york_coordinates, zoom_level=15)
markers = gmaps.marker_layer(marker_loc)
fig.add_layer(markers)
fig

# COMMAND ----------

model_output_df = spark.read.format("delta").load(MODEL_INFO)

# COMMAND ----------

display(model_output_df)

# COMMAND ----------

dbutils.widgets.dropdown("Promote Model","No",["No","Yes"])
dbutils.widgets.text("Hours to Forecast","8")

# COMMAND ----------

hours_to_forecast = int(dbutils.widgets.get("Hours to Forecast"))

# COMMAND ----------

model_output = model_output_df.toPandas()
model_output["capacity"] = 61

# COMMAND ----------

forecast_df = model_output.iloc[-hours_to_forecast:,:]

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
fig = px.line(forecast_df, x="ds", y="yhat", title='Forecast',markers=True)
fig.add_trace(go.Scatter(x=forecast_df.ds, y=forecast_df["capacity"], name='station capacity',
                         line = dict(color='blue', width=4, dash='dot')))
fig.show()

# COMMAND ----------

#plot the residuals
fig = px.scatter(
    forecast_df, x='yhat', y='residual',
    marginal_y='violin',
    trendline='ols'
)
fig.show()

# COMMAND ----------

client = MlflowClient()
ARTIFACT_PATH = GROUP_MODEL_NAME

# COMMAND ----------

latest_version_info = client.get_latest_versions(ARTIFACT_PATH, stages=["Staging"])

latest_staging_version = latest_version_info[0].version

print("The latest staging version of the model '%s' is '%s'." % (ARTIFACT_PATH, latest_staging_version))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 0,YOUR APPLICATIONS CODE HERE...
# start_date = str(dbutils.widgets.get('01.start_date'))
# end_date = str(dbutils.widgets.get('02.end_date'))
# hours_to_forecast = int(dbutils.widgets.get('03.hours_to_forecast'))
# promote_model = bool(True if str(dbutils.widgets.get('04.promote_model')).lower() == 'yes' else False)

# print(start_date,end_date,hours_to_forecast, promote_model)

# print("YOUR CODE HERE...")

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
