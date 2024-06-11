import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

start_time_rdd = time.time()

spark = (
    SparkSession.builder.appName("Crime Data Analysis Q2 RDD")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "2g")
    .config("spark.shuffle.io.netty.eventLoopThreads", "8")
    .getOrCreate()
)

spark.conf.set("spark.sql.shuffle.partitions", "8")

spark.sparkContext.setLogLevel("WARN")


crime_df = (
    spark.read.csv(
        "hdfs://master:9000/home/user/crime_data", header=True, inferSchema=True
    )
    .select(F.col("TIME OCC").cast(IntegerType()), F.col("Premis Desc"))
    .filter((F.col("Premis Desc") == "STREET") & (F.col("TIME OCC").isNotNull()))
    .sortWithinPartitions("TIME OCC")
)


street_crimes_rdd = crime_df.rdd


time_segments_rdd = (
    street_crimes_rdd.map(lambda row: (
        ("Morning" if 500 <= row["TIME OCC"] < 1200 else
         "Afternoon" if 1200 <= row["TIME OCC"] < 1700 else
         "Evening" if 1700 <= row["TIME OCC"] < 2100 else
         "Night"), 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: -x[1])
)


results = time_segments_rdd.collect()
results_df = spark.createDataFrame(results, ["Time Segment", "Count"])


results_df.show()

end_time_rdd = time.time()
print("RDD API Execution Time: {:.2f} seconds".format(end_time_rdd - start_time_rdd))
spark.stop()
