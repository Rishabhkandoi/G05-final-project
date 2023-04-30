# Databricks notebook source
# Import Statements

import pandas as pd
import requests
import json
import datetime
import time
import numpy as np

from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

GROUPS = [['G01', 'tcharle3@ur.rochester.edu', 'jbrehm@ur.rochester.edu', 'ccolli29@ur.rochester.edu', 'varvind@ur.rochester.edu', 'jwang247@ur.rochester.edu'],
['G02', 'gdumpa@ur.rochester.edu', 'vkakunur@ur.rochester.edu', 'nmalkann@ur.rochester.edu', 'hkotavet@ur.rochester.edu', 'sroy22@ur.rochester.edu'],
['G03', 'lnguy32@ur.rochester.edu', 'hvu7@ur.rochester.edu', 'btran10@u.rochester.edu', 'jperovi3@ur.rochester.edu', None],
['G04', 'tshroff@ur.rochester.edu', 'asingla3@ur.rochester.edu', 'prao13@ur.rochester.edu', 'htalele@ur.rochester.edu', 'hloya@ur.rochester.edu'],
['G05', 'rkandoi@ur.rochester.edu', 'amath12@ur.rochester.edu', 'ryadav3@ur.rochester.edu', None, None],
['G06', 'achilla@ur.rochester.edu', 'nkorrapo@ur.rochester.edu', 'dmunot@ur.rochester.edu', 'apalit@ur.rochester.edu', 'sgarg11@ur.rochester.edu'],
['G07', 'skishor2@ur.rochester.edu', 'ssingh71@ur.rochester.edu', 'sswar@ur.rochester.edu', 'dgarg@ur.rochester.edu', 'stamhane@ur.rochester.edu'],
['G08', 'ikaplan4@u.rochester.edu', 'mkingsl6@u.rochester.edu', 'agolli@u.rochester.edu', 'nboonin@u.rochester.edu', None],
['G09', 'jchen134@u.rochester.edu', 'qtang5@u.rochester.edu', 'jyon@u.rochester.edu', 'cmiao3@u.rochester.edu', 'zwang154@u.rochester.edu'],
['G10', 'zgu12@ur.rochester.edu', 'tluo9@ur.rochester.edu', 'dyu18@ur.rochester.edu', 'jcrosset@ur.rochester.edu', 'klee109@u.rochester.edu'],
['G11', 'vchistay@u.rochester.edu', 'fcolombo@u.rochester.edu', 'agirsky@u.rochester.edu', 'ajaweed@ur.rochester.edu', 'jtschopp@u.rochester.edu'],
['G12', 'nwang28@ur.rochester.edu', 'mwu52@ur.rochester.edu', 'clu36@ur.rochester.edu', 'aabbaraj@ur.rochester.edu', None],
['G13', 'akarunan@ur.rochester.edu', 'eklutse@ur.rochester.edu', 'snuamaha@ur.rochester.edu', 'tmurali@ur.rochester.edu', 'stodi@ur.rochester.edu'],
['GXX', 'lpalum@gmail.com', None, None, None, None]
]

GROUPS_STATION_ASSIGNMENT = {'G01':'W 21 St & 6 Ave',
 'G02':'West St & Chambers St',
 'G03':'1 Ave & E 68 St',
 'G04':'6 Ave & W 33 St',
 'G05':'University Pl & E 14 St',
 'G06':'Broadway & E 14 St',
 'G07':'Broadway & W 25 St',
 'G08':'W 31 St & 7 Ave',
 'G09':'E 33 St & 1 Ave',
 'G10':'8 Ave & W 33 St',
 'G11':'Cleveland Pl & Spring St',
 'G12':'11 Ave & W 41 St',
 'G13':'Lafayette St & E 8 St',
 'GXX':'Lafayette St & E 8 St'}

# COMMAND ----------

GROUP_STATION_ID = "b30815c0-99b6-451b-be15-902992cb8abb"

# COMMAND ----------

# DBTITLE 1,Define Project Global Variables
NYC_WEATHER_FILE_PATH = 'dbfs:/FileStore/tables/raw/weather/'
BIKE_TRIP_DATA_PATH = 'dbfs:/FileStore/tables/raw/bike_trips/'

BRONZE_STATION_INFO_PATH = 'dbfs:/FileStore/tables/bronze_station_info.delta'
BRONZE_STATION_STATUS_PATH = 'dbfs:/FileStore/tables/bronze_station_status.delta'
BRONZE_NYC_WEATHER_PATH = 'dbfs:/FileStore/tables/bronze_nyc_weather.delta'

# Some configuration of the cluster
spark.conf.set("spark.sql.shuffle.partitions", "40")  # Configure the size of shuffles the same as core count on your cluster
spark.conf.set("spark.sql.adaptive.enabled", "true")  # Spark 3.0 AQE - coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization


USER_NAME = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())['tags']['user']
GROUP_NAME = ""

for i in range(len(GROUPS)):
    if USER_NAME in GROUPS[i]:
        GROUP_NAME = GROUPS[i][0]

if GROUP_NAME == "":
    dbutils.notebook.exit(f"{USER_NAME} is  not assigned to a final project group")
    

GROUP_STATION_ASSIGNMENT = GROUPS_STATION_ASSIGNMENT[GROUP_NAME]

GROUP_MODEL_NAME = f"{GROUP_NAME}_model"

GROUP_DB_NAME = f"{GROUP_NAME}_db"

GROUP_DATA_PATH = f"dbfs:/FileStore/tables/{GROUP_NAME}/"


# Setup the hive meta store if it does not exist and select database as the focus of future sql commands in this notebook
spark.sql(f"CREATE DATABASE IF NOT EXISTS {GROUP_DB_NAME}")
spark.sql(f"USE {GROUP_DB_NAME}")

# COMMAND ----------

# Create directories

# Creating Bronze/Silver/Gold Driectories
BRONZE_DIR = GROUP_DATA_PATH + "bronze"
SILVER_DIR = GROUP_DATA_PATH + "silver"
GOLD_DIR = GROUP_DATA_PATH + "gold"

# Creating bronze storage structure
REAL_TIME_STATION_STATUS_DELTA_DIR = BRONZE_DIR + "/real_time_station_status"
REAL_TIME_STATION_INFO_DELTA_DIR = BRONZE_DIR + "/real_time_station_info"
REAL_TIME_WEATHER_DELTA_DIR = BRONZE_DIR + "/real_time_weather"
HISTORIC_STATION_INFO_DELTA_DIR = BRONZE_DIR + "/historic_station_info"
HISTORIC_WEATHER_DELTA_DIR = BRONZE_DIR + "/historic_weather"

# Creating checkpoints for bronze data
REAL_TIME_STATION_STATUS_CHECKPOINT_DIR = REAL_TIME_STATION_STATUS_DELTA_DIR + "/checkpoints"
REAL_TIME_STATION_INFO_CHECKPOINT_DIR = REAL_TIME_STATION_INFO_DELTA_DIR + "/checkpoints"
REAL_TIME_WEATHER_CHECKPOINT_DIR = REAL_TIME_WEATHER_DELTA_DIR + "/checkpoints"
HISTORIC_STATION_INFO_CHECKPOINT_DIR = HISTORIC_STATION_INFO_DELTA_DIR + "/checkpoints"
HISTORIC_WEATHER_CHECKPOINT_DIR = HISTORIC_WEATHER_DELTA_DIR + "/checkpoints"

# Creating Silver storage structure
REAL_TIME_INVENTORY_INFO_DELTA_DIR = SILVER_DIR + "/real_time_inventory_info"
HISTORIC_INVENTORY_INFO_DELTA_DIR = SILVER_DIR + "/historic_inventory_info"
INVENTORY_INFO_DELTA_DIR = SILVER_DIR + "/inventory_info"
WEATHER_INFO_DELTA_DIR = SILVER_DIR + "/weather_info"

# Creating checkpoints for silver data
REAL_TIME_INVENTORY_INFO_CHECKPOINT_DIR = REAL_TIME_INVENTORY_INFO_DELTA_DIR + "/checkpoints"
HISTORIC_INVENTORY_INFO_CHECKPOINT_DIR = HISTORIC_INVENTORY_INFO_DELTA_DIR + "/checkpoints"
INVENTORY_INFO_CHECKPOINT_DIR = INVENTORY_INFO_DELTA_DIR + "/checkpoints"
WEATHER_INFO_CHECKPOINT_DIR = WEATHER_INFO_DELTA_DIR + "/checkpoints"

# Creating Gold storage structure
MODEL_INFO = GOLD_DIR + "/model_info"

# Creating checkpoints for gold data
MODEL_CHECKPOINT_DIR = MODEL_INFO + "/checkpoints"

# Running mkdir for all above directories
dbutils.fs.mkdirs(BRONZE_DIR)
dbutils.fs.mkdirs(REAL_TIME_STATION_STATUS_DELTA_DIR)
dbutils.fs.mkdirs(REAL_TIME_STATION_INFO_DELTA_DIR)
dbutils.fs.mkdirs(REAL_TIME_WEATHER_DELTA_DIR)
dbutils.fs.mkdirs(HISTORIC_STATION_INFO_DELTA_DIR)
dbutils.fs.mkdirs(HISTORIC_WEATHER_DELTA_DIR)
dbutils.fs.mkdirs(REAL_TIME_STATION_STATUS_CHECKPOINT_DIR)
dbutils.fs.mkdirs(REAL_TIME_STATION_INFO_CHECKPOINT_DIR)
dbutils.fs.mkdirs(REAL_TIME_WEATHER_CHECKPOINT_DIR)
dbutils.fs.mkdirs(HISTORIC_STATION_INFO_CHECKPOINT_DIR)
dbutils.fs.mkdirs(HISTORIC_WEATHER_CHECKPOINT_DIR)

dbutils.fs.mkdirs(SILVER_DIR)
dbutils.fs.mkdirs(REAL_TIME_INVENTORY_INFO_DELTA_DIR)
dbutils.fs.mkdirs(HISTORIC_INVENTORY_INFO_DELTA_DIR)
dbutils.fs.mkdirs(REAL_TIME_INVENTORY_INFO_CHECKPOINT_DIR)
dbutils.fs.mkdirs(HISTORIC_INVENTORY_INFO_CHECKPOINT_DIR)
dbutils.fs.mkdirs(INVENTORY_INFO_DELTA_DIR)
dbutils.fs.mkdirs(INVENTORY_INFO_CHECKPOINT_DIR)

dbutils.fs.mkdirs(GOLD_DIR)
dbutils.fs.mkdirs(MODEL_INFO)
dbutils.fs.mkdirs(MODEL_CHECKPOINT_DIR)

# COMMAND ----------

# Adding our station info
STATION_LAT = 40.734814
STATION_LON = -73.992085
STATION_CAPACITY = 61

# Widget Inputs
dbutils.widgets.dropdown('Hours to Forecast', '48', ['4', '8', '12', '24', '48'])
dbutils.widgets.dropdown('Promote Model', 'No', ['No', 'Yes'])
HOURS_TO_FORECAST = int(dbutils.widgets.get('Hours to Forecast'))
PROMOTE_MODEL = bool(True if str(dbutils.widgets.get('Promote Model')).lower() == 'yes' else False)

# Model Tags
STAGING = 'Staging'
PROD = 'Production'
ARCHIVE = 'Archived'

# COMMAND ----------

# To enable inferring schema for historic data when reading with readStream
spark.conf.set("spark.sql.streaming.schemaInference", True)

# To disable format check while writing delta
spark.conf.set("spark.databricks.delta.formatCheck.enabled", False)

# COMMAND ----------

# DBTITLE 1,Display the Project Global Variables
displayHTML(f"""
<H1>VERY IMPORTANT TO UNDERSTAND THE USE OF THESE VARIABLES!<br> Please ask if you are confused about their use.</H1>
<table border=1>
<tr><td><b>Variable Name</b></td><td><b>Value</b></td><td><b>Description</b></td></tr>
<tr><td>NYC_WEATHER_FILE_PATH</td><td>{NYC_WEATHER_FILE_PATH}</td><td>Historic NYC Weather for Model Building</td></tr>
<tr><td>BIKE_TRIP_DATA_PATH</td><td>{BIKE_TRIP_DATA_PATH}</td><td>Historic Bike Trip Data for Model Building (Stream this data source)</td></tr>
<tr><td>BRONZE_STATION_INFO_PATH</td><td>{BRONZE_STATION_INFO_PATH}</td><td>Station Information (30 min refresh)</td></tr>
<tr><td>BRONZE_STATION_STATUS_PATH</td><td>{BRONZE_STATION_STATUS_PATH}</td><td>Station Status (30 min refresh)</td></tr>
<tr><td>BRONZE_NYC_WEATHER_PATH</td><td>{BRONZE_NYC_WEATHER_PATH}</td><td>NYC Weather (30 min refresh)</td></tr>
<tr><td>USER_NAME</td><td>{USER_NAME}</td><td>Email of the user executing this code/notebook</td></tr>
<tr><td>GROUP_NAME</td><td>{GROUP_NAME}</td><td>Group Assigment for this user</td></tr>
<tr><td>GROUP_STATION_ASSIGNMENT</td><td>{GROUP_STATION_ASSIGNMENT}</td><td>Station Name to be modeled by this group</td></tr>
<tr><td>GROUP_DATA_PATH</td><td>{GROUP_DATA_PATH}</td><td>Path to store all of your group data files (delta ect)</td></tr>
<tr><td>GROUP_MODEL_NAME</td><td>{GROUP_MODEL_NAME}</td><td>Mlflow Model Name to be used to register your model</td></tr>
<tr><td>BRONZE_DIR</td><td>{BRONZE_DIR}</td><td>Bronze Directory (Defined by us)</td></tr>
<tr><td>SILVER_DIR</td><td>{SILVER_DIR}</td><td>Silver Directory (Defined by us)</td></tr>
<tr><td>GOLD_DIR</td><td>{GOLD_DIR}</td><td>Gold Directory (Defined by us)</td></tr>
<tr><td>REAL_TIME_STATION_STATUS_DELTA_DIR</td><td>{REAL_TIME_STATION_STATUS_DELTA_DIR}</td><td>Streaming Station Status Delta Directory (Defined by us)</td></tr>
<tr><td>REAL_TIME_STATION_INFO_DELTA_DIR</td><td>{REAL_TIME_STATION_INFO_DELTA_DIR}</td><td>Streaming Station Info Delta Directory (Defined by us)</td></tr>
<tr><td>REAL_TIME_WEATHER_DELTA_DIR</td><td>{REAL_TIME_WEATHER_DELTA_DIR}</td><td>Streaming Weather Info Delta Directory (Defined by us)</td></tr>
<tr><td>HISTORIC_STATION_INFO_DELTA_DIR</td><td>{HISTORIC_STATION_INFO_DELTA_DIR}</td><td>Historic Station Info Delta Directory (Defined by us)</td></tr>
<tr><td>HISTORIC_WEATHER_DELTA_DIR</td><td>{HISTORIC_WEATHER_DELTA_DIR}</td><td>Historic Weather Info Delta Directory (Defined by us)</td></tr>
<tr><td>REAL_TIME_INVENTORY_INFO_DELTA_DIR</td><td>{REAL_TIME_INVENTORY_INFO_DELTA_DIR}</td><td>Streaming Inventory Info (after transformation) Delta Directory (Defined by us)</td></tr>
<tr><td>HISTORIC_INVENTORY_INFO_DELTA_DIR</td><td>{HISTORIC_INVENTORY_INFO_DELTA_DIR}</td><td>Historic Inventory Info (after transformation) Delta Directory (Defined by us)</td></tr>
<tr><td>INVENTORY_INFO_DELTA_DIR</td><td>{INVENTORY_INFO_DELTA_DIR}</td><td>Merged Inventory Info (after transformation) Delta Directory (Defined by us)</td></tr>
<tr><td>MODEL_INFO</td><td>{MODEL_INFO}</td><td>Model Info Delta Directory (Defined by us)</td></tr>
<tr><td>STATION_LAT</td><td>{STATION_LAT}</td><td>Group Station Latitude (Defined by us)</td></tr>
<tr><td>STATION_LON</td><td>{STATION_LON}</td><td>Group Station Longitude (Defined by us)</td></tr>
<tr><td>STATION_CAPACITY</td><td>{STATION_CAPACITY}</td><td>Group Station Capacity (Defined by us)</td></tr>
<tr><td>HOURS_TO_FORECAST</td><td>{HOURS_TO_FORECAST}</td><td>Widget Input for hours to forecast (Defined by us)</td></tr>
<tr><td>PROMOTE_MODEL</td><td>{PROMOTE_MODEL}</td><td>Widget Input for promote model (Defined by us)</td></tr>
<tr><td>STAGING</td><td>{STAGING}</td><td>Staging tag (Defined by us)</td></tr>
<tr><td>PROD</td><td>{PROD}</td><td>Production tag (Defined by us)</td></tr>
<tr><td>ARCHIVE</td><td>{ARCHIVE}</td><td>Archive tag (Defined by us)</td></tr>
</table>
""")
