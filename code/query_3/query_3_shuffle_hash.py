import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, DoubleType
from pyspark.sql.functions import col, to_timestamp, broadcast, when


spark = (
    SparkSession.builder.appName("Crime Data Analysis Q3")
    .config("spark.sql.debug.maxToStringFields", "1000")
    .config("spark.executor.memory", "2g")
    .getOrCreate()
)


spark.conf.set("spark.sql.shuffle.partitions", "4")

spark.sparkContext.setLogLevel("WARN")


income_schema = "ZIP_CODE STRING, Community STRING, Estimated_Median_Income STRING"
income_df = spark.read.csv(
    "hdfs://master:9000/home/user/Big_Data/LA_income_2015.csv",
    header=True,
    schema=income_schema,
)


income_df = income_df.withColumn(
    "Estimated_Median_Income",
    F.regexp_replace(col("Estimated_Median_Income"), "[$,]", "").cast(IntegerType())).cache()




geocoding_schema = StructType([
    StructField("LAT", DoubleType(), True),
    StructField("LON", DoubleType(), True),
    StructField("ZIPcode", StringType(), True)
])


geocoding_df = (
    spark.read.csv(
        "hdfs://master:9000/home/user/Big_Data/revgecoding.csv",
        header=True,
        schema=geocoding_schema,
    )
    .withColumn("ZIP_CODE", F.split(col("ZIPcode"), ";").getItem(0))
    .drop(F.col("ZIPcode"))
)



crime_2015_df = (
    spark.read.parquet("hdfs://master:9000/home/user/crime_data_parquet_2015")
    .filter(F.col("Vict Descent").isNotNull())
    .filter(F.col("Vict Descent") != "X")
    .select("LAT", "LON", "Vict Descent")
    .withColumn("LAT", F.col("LAT").cast("double"))
    .withColumn("LON", F.col("LON").cast("double"))
)


crime_with_zip_df = crime_2015_df.hint("shuffle_hash").join(
    geocoding_df.hint("shuffle_hash"),
    (crime_2015_df.LAT == geocoding_df.LAT) & (crime_2015_df.LON == geocoding_df.LON),
    "left",
).select(F.col("Vict Descent"), F.col("ZIP_CODE")).cache()



zip_codes = [
     row["ZIP_CODE"] for row in crime_with_zip_df.select("ZIP_CODE").distinct().collect()if row["ZIP_CODE"] is not None
]


top_3_income_zips = [
    row["ZIP_CODE"]
    for row in income_df.filter(col("ZIP_CODE").isin(zip_codes))
    .orderBy(col("Estimated_Median_Income").desc())
    .limit(3)
    .collect()
]


bottom_3_income_zips = [
    row["ZIP_CODE"]
    for row in income_df.filter(col("ZIP_CODE").isin(zip_codes))
    .orderBy(col("Estimated_Median_Income"))
    .limit(3)
    .collect()
]


top_3_income_crimes_df = crime_with_zip_df.filter(
    col("ZIP_CODE").isin(top_3_income_zips)
)
bottom_3_income_crimes_df = crime_with_zip_df.filter(
    col("ZIP_CODE").isin(bottom_3_income_zips)
)


top_3_income_victims = (
    top_3_income_crimes_df.groupBy("Vict Descent").count().orderBy(col("count").desc())
).withColumnRenamed("count", "total victims")

bottom_3_income_victims = (
    bottom_3_income_crimes_df.groupBy("Vict Descent")
    .count()
    .orderBy(col("count").desc())
).withColumnRenamed("count", "total victims")


mapping_expr = when(col("Vict Descent") == "A", "Asian") \
    .when(col("Vict Descent") == "B", "Black") \
    .when(col("Vict Descent") == "C", "Chinese") \
    .when(col("Vict Descent") == "D", "Cambodian") \
    .when(col("Vict Descent") == "F", "Filipino") \
    .when(col("Vict Descent") == "G", "Guamanian") \
    .when(col("Vict Descent") == "H", "Hispanic/Latin/Mexican") \
    .when(col("Vict Descent") == "I", "American Indian/Alaskan Native") \
    .when(col("Vict Descent") == "J", "Japanese") \
    .when(col("Vict Descent") == "K", "Korean") \
    .when(col("Vict Descent") == "L", "Laotian") \
    .when(col("Vict Descent") == "O", "Other") \
    .when(col("Vict Descent") == "P", "Pacific Islander") \
    .when(col("Vict Descent") == "S", "Samoan") \
    .when(col("Vict Descent") == "U", "Hawaiian") \
    .when(col("Vict Descent") == "V", "Vietnamese") \
    .when(col("Vict Descent") == "W", "White") \
    .when(col("Vict Descent") == "X", "Unknown") \
    .when(col("Vict Descent") == "Z", "Asian Indian") \
    .otherwise("Unknown")


top_3_income_victims = top_3_income_victims.withColumn("Vict Descent", mapping_expr)
bottom_3_income_victims = bottom_3_income_victims.withColumn("Vict Descent", mapping_expr)


print("Top 3 Income ZIP Codes Victims by Descent")
top_3_income_victims.show(10, False)

print("Bottom 3 Income ZIP Codes Victims by Descent")
bottom_3_income_victims.show(10, False)


spark.stop()
