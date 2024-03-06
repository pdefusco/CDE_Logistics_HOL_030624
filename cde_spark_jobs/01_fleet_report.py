#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import sys, random, os, json, random, configparser
from utils import *

spark = SparkSession \
    .builder \
    .appName("IOT FLEET BATCH REPORT") \
    .getOrCreate()

config = configparser.ConfigParser()
config.read('/app/mount/parameters.conf')
storageLocation=config.get("general","data_lake_name")
print("Storage Location from Config File: ", storageLocation)

username = sys.argv[1]
print("PySpark Runtime Arg: ", sys.argv[1])

### FIRST BATCH FACT TABLE

firstBatchDf = spark.read.json("{0}/logistics/firstbatch/{1}/iotfleet".format(storageLocation, username))
firstBatchDf = firstBatchDf.select(flatten_struct(firstBatchDf.schema))
firstBatchDf.printSchema()

### RENAME MULTIPLE COLUMNS
cols = [col for col in firstBatchDf.columns if col.startswith("iot_geolocation")]
new_cols = [col.split(".")[1] for col in cols]
firstBatchDf = renameMultipleColumns(firstBatchDf, cols, new_cols)

### CAST TYPES
cols = ["latitude", "longitude"]
firstBatchDf = castMultipleColumns(firstBatchDf, cols)
firstBatchDf = firstBatchDf.withColumn("event_ts", firstBatchDf["event_ts"].cast("timestamp"))

### TRX DF SCHEMA AFTER CASTING AND RENAMING
firstBatchDf.printSchema()

### STORE FIRST BATCH AS TABLE
spark.sql("DROP DATABASE IF EXISTS {} CASCADE".format(username))
spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(username))
spark.sql("SHOW DATABASES LIKE '{}'".format(username)).show()
firstBatchDf.write.mode("overwrite").saveAsTable('{}.FIRST_BATCH_TABLE'.format(username), format="parquet")

### PII DIMENSION TABLE
companyDf = spark.read.options(header='True', delimiter=',').csv("{0}/logistics/company/{1}/company_info".format(storageLocation, username))

### CAST LAT LON AS FLOAT
companyDf = companyDf.withColumn("facility_latitude",  companyDf["facility_latitude"].cast('float'))
companyDf = companyDf.withColumn("facility_longitude",  companyDf["facility_longitude"].cast('float'))

### STORE COMPANY DATA AS TABLE
companyDf.write.mode("overwrite").saveAsTable('{}.COMPANY_TABLE'.format(username), format="parquet")

### JOIN TWO DATASETS AND COMPARE COORDINATES
joinDf = spark.sql("""SELECT iot.device_id, iot.event_type, iot.event_ts, iot.latitude, iot.longitude, iot.iot_signal_1,
          iot.iot_signal_2, iot.iot_signal_3, iot.iot_signal_4, i.company_name, i.company_email, i.facility_latitude, i.facility_longitude
          FROM company_info i INNER JOIN firstbatch iot
          ON i.manufacturer == iot.manufacturer;""".format(username))

print("JOINDF SCHEMA")
joinDf.printSchema()

### PANDAS UDF
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

# Method for Euclidean Distance
def euclidean_dist(x1: pd.Series, x2: pd.Series, y1: pd.Series, y2: pd.Series) -> pd.Series:
   return ((x2-x1)**2)+((y2-y1)**2).pow(1./2)

# Saving Method as Pandas UDF
eu_dist = pandas_udf(euclidean_dist, returnType=FloatType())

# Applying UDF on joinDf
eucDistDf = joinDf.withColumn("DIST_FROM_FACILITY", eu_dist(F.col("facility_longitude"), \
                                      F.col("longitude"), F.col("facility_latitude"), \
                                       F.col("latitude")))

# SELECT CUSTOMERS WHERE FLEET UNIT IS LOCATED MORE THAN 20 MILES FROM HOME
eucDistDf.filter(eucDistDf.DIST_FROM_FACILITY > 20).show()
