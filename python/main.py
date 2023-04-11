import json
import datetime

from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

from datetime import datetime
from pyspark.sql import SparkSession

#===================================================================
# Spark
#===================================================================
spark = SparkSession.builder \
            .appName("Case Krowdy") \
            .config("spark.driver.maxResultSize","8G") \
            .config("spark.driver.cores", 4) \
            .config("spark.driver.memory", "8G") \
            .config("spark.executor.cores", 8) \
            .config("spark.executor.memory", "8G") \
            .config("spark.executor.pyspark.memory", "8G") \
            .config("spark.sql.shuffle.partitions", 1000) \
            .getOrCreate()

#===================================================================
# Variables Configuration
#===================================================================

path_csv = "../files/2023-04-11T04-36-02-388Zinstituciones_educativas.csv"
path_json = "../files/2023-04-11T04-36-09-824ZUniversidades.json"

#===================================================================
# Fuctions
#===================================================================

def read_csv(path_file):
    return(spark.read.option("header",True).csv(path_file))

def read_json(path_file):
    return(spark.read.option("multiline","true").option("mode", "PERMISSIVE").format("json").load(path_file))

#===================================================================

def  process_df():
    df_csv = read_csv(path_csv)
    df_json = read_json(path_json)

    df_csv.show(truncate = False)
    df_json.show(truncate = False)

    print(df_csv.count())
    print(df_json.count())


#===================================================================

def main():
    process_df()
    



if __name__ == '__main__':
    t1 = datetime.now()
    print('started at:', t1)
    
    main()

    t2 = datetime.now()
    dist = t2 - t1
    print('finished at:', t2, ' | elapsed time (s):', dist.seconds)