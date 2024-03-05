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

import os
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from datetime import datetime
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, company
from pyspark.sql.types import LongType, FloatType, IntegerType, StringType, \
                              DoubleType, BooleanType, ShortType, \
                              TimestampType, DateType, DecimalType, \
                              ByteType, BinaryType, ArrayType, MapType, \
                              StructType, StructField

class IotDataGen:

    '''Class to Generate IoT Fleet Data'''

    def __init__(self, spark):
        self.spark = spark


    def firstBatchDataGen(self, minLatitude, maxLatitude, minLongitude, maxLongitude, shuffle_partitions_requested = 2, partitions_requested = 5, data_rows = 14400):
        """
        Method to create credit card transactions in Spark Df
        """

        manufacturers = ["New World Corp", "AIAI Inc.", "Hot Data Ltd"]

        iotDataSpec = (
            dg.DataGenerator(self.spark, name="device_data_set", rows=data_rows, partitions=partitions_requested)
            .withIdOutput()
            .withColumn("internal_device_id", "long", minValue=0x1000000000000, uniqueValues=int(data_rows/1440), omit=True, baseColumnType="hash", randomSeed=4)
            .withColumn("device_id", "string", format="0x%013x", baseColumn="internal_device_id")
            .withColumn("manufacturer", "string", values=manufacturers, baseColumn="internal_device_id", )
            .withColumn("model_ser", "integer", minValue=1, maxValue=11, baseColumn="device_id", baseColumnType="hash", omit=True, )
            .withColumn("event_type", "string", values=["tank below 10%", "tank below 5%", "device error", "system malfunction"], random=True)
            .withColumn("event_ts", "timestamp", begin="2023-12-01 01:00:00", end="2023-12-01 23:59:00", interval="1 minute", random=False )
            .withColumn("longitude", "float", minValue=minLongitude, maxValue=maxLongitude, random=True )
            .withColumn("latitude", "float", minValue=minLatitude, maxValue=maxLatitude, random=True )
            .withColumn("iot_signal_1", "integer", minValue=1, maxValue=10, random=True)
            .withColumn("iot_signal_2", "integer", minValue=1000, maxValue=1020, random=True)
            .withColumn("iot_signal_3", "integer", minValue=50, maxValue=55, random=True)
            .withColumn("iot_signal_4", "integer", minValue=100, maxValue=107, random=True)
            .withColumn("transaction_geolocation", StructType([StructField('latitude',StringType()), StructField('longitude', StringType())]),
              expr="named_struct('latitude', latitude, 'longitude', longitude)",
              baseColumn=['latitude', 'longitude'])
        )

        df = iotDataSpec.build()

        df = df.drop(*('latitude', 'longitude'))

        return df


    def secondBatchDataGen(self, minLatitude, maxLatitude, minLongitude, maxLongitude, shuffle_partitions_requested = 1, partitions_requested = 1, data_rows = 1000):
        """
        Method to create credit card transactions in Spark Df
        """

        manufacturers = ["New World Corp", "AIAI Inc.", "Hot Data Ltd"]

        iotDataSpec = (
            dg.DataGenerator(self.spark, name="device_data_set", rows=data_rows, partitions=partitions_requested)
            .withIdOutput()
            .withColumn("internal_device_id", "long", minValue=0x1000000000000, uniqueValues=int(data_rows), omit=True, baseColumnType="hash")
            .withColumn("device_id", "string", format="0x%013x", baseColumn="internal_device_id")
            .withColumn("manufacturer", "string", values=manufacturers, baseColumn="internal_device_id", )
            .withColumn("model_ser", "integer", minValue=1, maxValue=11, baseColumn="device_id", baseColumnType="hash", omit=True, )
            .withColumn("event_type", "string", values=["tank below 10%", "tank below 5%", "device error", "system malfunction"], random=True)
            .withColumn("event_ts", "timestamp", begin="2023-12-01 01:00:00", end="2023-12-02 23:59:00", interval="1 minute", random=True )
            .withColumn("longitude", "float", minValue=minLongitude, maxValue=maxLongitude, random=True )
            .withColumn("latitude", "float", minValue=minLatitude, maxValue=maxLatitude, random=True )
            .withColumn("iot_signal_1", "integer", minValue=1, maxValue=10, random=True)
            .withColumn("iot_signal_2", "integer", minValue=1000, maxValue=1020, random=True)
            .withColumn("iot_signal_3", "integer", minValue=50, maxValue=55, random=True)
            .withColumn("iot_signal_4", "integer", minValue=100, maxValue=107, random=True)
        )

        df = iotDataSpec.build()

        return df


    def thirdBatchDataGen(self, minLatitude, maxLatitude, minLongitude, maxLongitude, shuffle_partitions_requested = 1, partitions_requested = 1, data_rows = 1000):
        """
        Method to create credit card transactions in Spark Df
        """

        manufacturers = ["New World Corp", "AIAI Inc.", "Hot Data Ltd"]

        iotDataSpec = (
            dg.DataGenerator(self.spark, name="device_data_set", rows=data_rows, partitions=partitions_requested)
            .withIdOutput()
            .withColumn("internal_device_id", "long", minValue=0x1000000000000, uniqueValues=int(data_rows), omit=True, baseColumnType="hash")
            .withColumn("device_id", "string", format="0x%013x", baseColumn="internal_device_id")
            .withColumn("manufacturer", "string", values=manufacturers, baseColumn="internal_device_id", )
            .withColumn("model_ser", "integer", minValue=1, maxValue=11, baseColumn="device_id", baseColumnType="hash", omit=True, )
            .withColumn("event_type", "string", values=["tank below 10%", "tank below 5%", "device error", "system malfunction"], random=True)
            .withColumn("event_ts", "timestamp", begin="2023-12-03 01:00:00", end="2023-12-03 23:59:00", interval="1 minute", random=True )
            .withColumn("longitude", "float", minValue=minLongitude, maxValue=maxLongitude, random=True )
            .withColumn("latitude", "float", minValue=minLatitude, maxValue=maxLatitude, random=True )
            .withColumn("iot_signal_1", "integer", minValue=1, maxValue=10, random=True)
            .withColumn("iot_signal_2", "integer", minValue=1000, maxValue=1020, random=True)
            .withColumn("iot_signal_3", "integer", minValue=50, maxValue=55, random=True)
            .withColumn("iot_signal_4", "integer", minValue=100, maxValue=107, random=True)
        )

        df = iotDataSpec.build()

        return df

    def companyDataGen(self, shuffle_partitions_requested = 1, partitions_requested = 1, data_rows = 3):

        manufacturers = ["New World Corp", "AIAI Inc.", "Hot Data Ltd"]

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        self.spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(self.spark, rows=data_rows, partitions=partitions_requested)
                    .withColumn("manufacturer", "string", values=manufacturers, random=False, randomSeed=4)
                    .withColumn("company_name", text=FakerTextUS("company"), randomSeed=4)
                    .withColumn("company_email", text=FakerTextUS("ascii_company_email"), randomSeed=4)
                    )

        df = fakerDataspec.build()
        df = df.withColumn('facility_latitude', lit(41.5868))
        df = df.withColumn('facility_longitude', lit(93.6250))

        return df


    def createDatabase(self):
        """
        Method to create database before data generated is saved to new database and table
        """

        self.spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        self.spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def dropDatabase(self):
        """
        Method to drop database used by previous demo run
        """

        print("SHOW DATABASES PRE DROP")
        self.spark.sql("SHOW DATABASES").show()
        self.spark.sql("DROP DATABASE IF EXISTS {} CASCADE;".format(self.dbname))
        print("\nSHOW DATABASES AFTER DROP")
        self.spark.sql("SHOW DATABASES").show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the IOT DEVICES FLEET table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.IOT_FLEET_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.IOT_FLEET_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        self.spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()
