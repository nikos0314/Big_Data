import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

start_time = time.time()

spark = (
	SparkSession
	.builder 
    	.appName("Crime Data Analysis DF parquet") 
    	.config("spark.executor.memory", "2g") 
	.config("spark.sql.debug.maxToStringFields", "1000") 
	.getOrCreate()
)

spark.conf.set("spark.sql.shuffle.partitions", "8")
spark.sparkContext.setLogLevel("WARN")


crime_df = (
    spark.read.parquet("hdfs://master:9000/home/user/Big_Data/crime_data_parquet.snappy.parquet", header=True)
    .withColumn("DATE OCC", F.to_timestamp(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))
    .select(
        F.year(F.col("DATE OCC")).alias("year"),
        F.month(F.col("DATE OCC")).alias("month"),
    )
)



window_spec = Window.partitionBy("year").orderBy(F.desc("crime_total"))

crime_monthly_df = (
    crime_df.groupBy(F.col("year"), F.col("month"))
    .agg(F.count("*").alias("crime_total"))
    .withColumn("ranking", F.rank().over(window_spec))
    .where(F.col("ranking") <= 3)
    .orderBy(F.col("year"), F.desc("crime_total"))
)

crime_monthly_df.show(50)

end_time = time.time()

print("DataFrame API Execution Time: {:.2f} seconds".format(end_time - start_time))

spark.stop()
