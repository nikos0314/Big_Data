import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

start_time = time.time()

spark = (
	SparkSession
	.builder 
    	.appName("Crime Data Analysis SQL parquet") 
    	.config("spark.executor.memory", "1g") 
	.config("spark.sql.debug.maxToStringFields", "1000") 
	.getOrCreate()
)

spark.conf.set("spark.sql.shuffle.partitions", "8")

spark.sparkContext.setLogLevel("WARN")


crime_df = (
    spark.read.parquet("hdfs://master:9000/home/user/crime_data_parquet")
    .withColumn("DATE OCC", F.to_timestamp(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))
    .select(
        F.year(F.col("DATE OCC")).alias("year"),
        F.month(F.col("DATE OCC")).alias("month"),
    )
)
crime_df.createOrReplaceTempView("crime_data")




crime_monthly_df = spark.sql(
    """
    SELECT year, month, crime_total, ranking FROM (
        SELECT
            year,
            month,
            COUNT(*) AS crime_total,
            RANK() OVER (PARTITION BY year ORDER BY COUNT(*) DESC) AS ranking
        FROM crime_data
        GROUP BY year, month
    ) WHERE ranking <= 3
    ORDER BY year, crime_total DESC
"""
)

crime_monthly_df.show(50)


end_time = time.time()


print("SQL Execution Time: {:.2f} seconds".format(end_time - start_time))

spark.stop()





