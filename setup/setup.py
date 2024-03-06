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

from os.path import exists
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from utils import *
from datetime import datetime
import sys, random, os, json, random, configparser

## CDE PROPERTIES

def parseProperties():
    """
    Method to parse total number of HOL participants argument
    """
    try:
        print("PARSING JOB ARGUMENTS...")
        maxParticipants = sys.argv[1]
        storageLocation = sys.argv[2]
    except Exception as e:
        print("READING JOB ARG UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return maxParticipants, storageLocation


def createSparkSession():
    """
    Method to create an Iceberg Spark Session
    """

    try:
        spark = SparkSession \
            .builder \
            .appName("LOGISTICS IOT FLEET LOAD") \
            .getOrCreate()
    except Exception as e:
        print("LAUNCHING SPARK SESSION UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return spark


def createFirstBatchData(spark):
    """
    Method to create first logistics batch dataframe using the dbldatagen and Faker frameworks
    """

    try:
        print("CREATING IOT FLEET FIRST BATCH DF...\n")
        dg = IotDataGen(spark)
        desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude = 40.51341, 43.67468, -94.9000, -92.5000
        firstBatchDf = dg.firstBatchDataGen(desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude)
    except Exception as e:
        print("CREATING FIRST BATCH UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return firstBatchDf


def createSecondBatchData(spark):
    """
    Method to create second logistics batch dataframe using the dbldatagen and Faker frameworks
    """

    try:
        print("CREATING IOT FLEET SECOND BATCH DF...\n")
        dg = IotDataGen(spark)
        desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude = 40.51341, 43.67468, -94.9000, -92.5000
        secondBatchDf = dg.secondBatchDataGen(desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude)
    except Exception as e:
        print("CREATING SECOND BATCH UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return secondBatchDf


def createThirdBatchData(spark):
    """
    Method to create third logistics batch dataframe using the dbldatagen and Faker frameworks
    """

    try:
        print("CREATING IOT FLEET THIRD BATCH DF...\n")
        dg = IotDataGen(spark)
        desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude = 40.51341, 43.67468, -94.9000, -92.5000
        thirdBatchDf = dg.thirdBatchDataGen(desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude)
    except Exception as e:
        print("CREATING TRANSACTION DATA 2 UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return thirdBatchDf


def createCompanyData(spark):
    """
    Method to create a trucking company dataframe using the dbldatagen and Faker frameworks
    """

    try:
        print("CREATING TRUCK COMPANY DF...\n")
        dg = IotDataGen(spark)
        companyDf = dg.companyDataGen()
    except Exception as e:
        print("CREATING COMPANY DATA UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    return companyDf


def saveFirstBatchData(firstBatchDf, storageLocation, username):
    """
    Method to save first batch data to Cloud Storage in Json format
    """

    print("SAVING FIRST BATCH OF IOT FLEET DATA TO JSON IN CLOUD STORAGE...\n")

    try:
        firstBatchDf. \
            write. \
            format("json"). \
            mode("overwrite"). \
            save("{0}/logistics/firstbatch/{1}/iotfleet".format(storageLocation, username))
    except Exception as e:
        print("SAVING FIRST BATCH DATA UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)


def saveSecondBatchData(secondBatchDf, storageLocation, username):
    """
    Method to save second batch data to Cloud Storage in Json format
    """

    print("SAVING SECOND BATCH OF IOT FLEET DATA TO JSON IN CLOUD STORAGE...\n")

    try:
        secondBatchDf. \
            write. \
            format("json"). \
            mode("overwrite"). \
            save("{0}/logistics/secondbatch/{1}/iotfleet".format(storageLocation, username))
    except Exception as e:
        print("SAVING SECOND BATCH DATA UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)


def saveThirdBatchData(thirdBatchDf, storageLocation, username):
    """
    Method to save third batch data to Cloud Storage in Json format
    """

    print("SAVING THIRD BATCH OF IOT FLEET DATA TO JSON IN CLOUD STORAGE...\n")

    try:
        thirdBatchDf. \
            write. \
            format("json"). \
            mode("overwrite"). \
            save("{0}/logistics/thirdbatch/{1}/iotfleet".format(storageLocation, username))
    except Exception as e:
        print("SAVING THIRD BATCH DATA UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)


def saveCompanyData(companyDf, storageLocation, username):
    """
    Method to save company information to Cloud Storage in csv format
    """

    print("SAVING COMPANY DF TO CSV IN CLOUD STORAGE...\n")

    try:
        companyDf \
            .write. \
            mode('overwrite') \
            .options(header='True', delimiter=',') \
            .csv("{0}/logistics/company/{1}/company_info".format(storageLocation, username))
    except Exception as e:
        print("SAVING COMPANY INFO UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)


def dropDatabase(spark, username):
    """
    Method to drop database used by previous demo run
    """

    print("SHOW DATABASES PRE DROP")
    spark.sql("SHOW DATABASES").show()
    spark.sql("DROP DATABASE IF EXISTS {} CASCADE;".format(username))
    print("\nSHOW DATABASES AFTER DROP")
    spark.sql("SHOW DATABASES").show()


def main():

    maxParticipants, storageLocation = parseProperties()

    spark = createSparkSession()

    for i in range(int(maxParticipants)):
        if i+1 < 10:
            username = "user00" + str(i+1)
        elif i+1 > 9 and i+1 < 99:
            username = "user0" + str(i+1)
        elif i+1 > 99:
            username = "user" + str(i+1)

        print("PROCESSING USER {}...\n".format(username))

        dropDatabase(spark, username)

        firstBatchDf = createFirstBatchData(spark)
        saveFirstBatchData(firstBatchDf, storageLocation, username)

        secondBatchDf = createSecondBatchData(spark)
        saveSecondBatchData(secondBatchDf, storageLocation, username)

        thirdBatchDf = createThirdBatchData(spark)
        saveThirdBatchData(thirdBatchDf, storageLocation, username)

        companyDf = createCompanyData(spark)
        saveCompanyData(companyDf, storageLocation, username)


if __name__ == "__main__":
    main()
