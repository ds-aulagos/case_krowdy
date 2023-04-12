import json
import datetime

from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

from datetime import datetime
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, NGram, HashingTF, MinHashLSH, RegexTokenizer, SQLTransformer

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

path_csv = "./files/2023-04-11T04-36-02-388Zinstituciones_educativas.csv"
path_json = "./files/2023-04-11T04-36-09-824ZUniversidades.json"

locale = spark._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("es-ES"))

sql_transformer = SQLTransformer(statement="SELECT *, lower(universidade) lower FROM __THIS__")
tokenizer = Tokenizer(inputCol="lower", outputCol="token")
stop_words_remover = StopWordsRemover(inputCol="token", outputCol="stop")
sql_transformer_concat = SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__")
regex_tokenizer = RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=1)
n_gram = NGram(n=2, inputCol="char", outputCol="ngram")
hashing_tf = HashingTF(inputCol="ngram", outputCol="vector")
min_hash_lsh = MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)

#===================================================================
# Fuctions
#===================================================================

def read_csv(path_file):
    return(spark.read.option("header",True).option("quote", "\"").csv(path_file))

def read_json(path_file):
    return(spark.read.option("multiline","true").option("mode", "PERMISSIVE").format("json").load(path_file))

#===================================================================

def  process_df():
    df_csv = read_csv(path_csv)\
                .withColumn("candidate_id", F.col("candidateId"))\
                .withColumn("universidade", F.regexp_replace(F.col("value"), "\"", ""))\
                .select("candidate_id","universidade")

    df_json = read_json(path_json)\
                .withColumn("cod_inei", F.col("código INEI"))\
                .withColumn("universidade", F.col("Nombre "))\
                .withColumn("siglas", F.col("Siglas "))\
                .select("universidade","siglas","cod_inei")

    df_value_siglas_lower = df_json.alias("j")\
                                .join(df_csv.alias("c"), [F.lower("j.siglas") == F.lower("c.universidade")], "inner")\
                                .select("*")

    df_csv_diff = df_csv.join(df_value_siglas_lower, ["universidade"], "left_anti").distinct()
    df_json_diff = df_json.join(df_value_siglas_lower, ["siglas"], "left_anti").distinct()

    pipeline = Pipeline(stages=[sql_transformer, tokenizer, stop_words_remover, sql_transformer_concat, regex_tokenizer, n_gram, hashing_tf, min_hash_lsh]).fit(df_json_diff)

    result_csv = pipeline.transform(df_csv_diff)
    result_csv = result_csv.filter(F.size(F.col("ngram")) > 0)
    #result_csv.select('candidate_id', 'universidade', 'concat', 'char', 'ngram', 'vector', 'lsh').show()

    result_json = pipeline.transform(df_json_diff)
    result_json = result_json.filter(F.size(F.col("ngram")) > 0)    
    #result_json.select('cod_inei', 'universidade', 'concat', 'char', 'ngram', 'vector', 'lsh').show()

    result = pipeline.stages[-1].approxSimilarityJoin(result_csv, result_json, 0.5, "jaccardDist")

    #result.select('datasetA.candidate_id', 'datasetA.universidade', 'datasetB.universidade', 'jaccardDist').sort(F.col('datasetA.candidate_id')).show(truncate = False)

    w = Window.partitionBy('datasetA.candidate_id')

    result = result\
                .withColumn('minDist', F.min('jaccardDist').over(w))\
                .where(F.col('jaccardDist') == F.col('minDist'))\
                .drop('minDist')

    df_output_csv = result.select('datasetA.candidate_id', F.col('datasetA.universidade').alias('value'), F.col('datasetB.universidade').alias('universidade homologada')).toPandas()

    df_output_csv.to_csv("./output/universidades_homologadas.csv")

    

#===================================================================

if __name__ == '__main__':
    t1 = datetime.now()
    print('started at:', t1)
    
    process_df()

    t2 = datetime.now()
    dist = t2 - t1
    print('finished at:', t2, ' | elapsed time (s):', dist.seconds)