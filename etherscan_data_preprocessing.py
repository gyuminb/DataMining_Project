import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, concat_ws, posexplode, count, array_distinct

def create_spark_session(app_name="Etherscan Data Processing"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_and_filter_data(spark, input_file):
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    return df.filter(col("Name Tag").isNull())

def create_item_column(df):
    return df.withColumn("Item", concat_ws("_", col("Chain"), col("Token"), col("Contract Address")))

def group_items_by_address(df):
    return df.groupBy("Address").agg(collect_list("Item").alias("Items"))

def find_duplicate_items(df):
    exploded_df = df.select("Address", posexplode("Items").alias("pos", "Item"))
    duplicate_items = exploded_df.groupBy("Address", "Item").agg(
        count("pos").alias("count"),
        collect_list("pos").alias("positions")
    )
    return duplicate_items.filter(col("count") > 1).withColumn("positions", concat_ws(",", "positions"))

def remove_duplicates_in_buckets(df):
    return df.withColumn("Items", array_distinct("Items"))

def save_dataframe_as_single_csv(df, output_file, temp_dir):
    df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")
    temp_file = [f for f in os.listdir(temp_dir) if f.startswith("part-")][0]
    shutil.move(os.path.join(temp_dir, temp_file), output_file)
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Step 1: Initialize Spark session
    spark = create_spark_session()

    # Step 2: Load data and filter for individual wallets only
    input_file = "etherscan_merged_data.csv"
    df_filtered = load_and_filter_data(spark, input_file)

    # Step 3: Create (Chain, Token, Contract Address) item representation
    df_items = create_item_column(df_filtered)

    # Step 4: Group items by wallet address
    df_buckets = group_items_by_address(df_items)

    # Step 5: Identify and save duplicate items
    df_duplicates = find_duplicate_items(df_buckets)
    save_dataframe_as_single_csv(df_duplicates, "duplicate_items.csv", "duplicates_temp")

    # Step 6: Remove duplicates from each bucket and save
    df_buckets_deduped = remove_duplicates_in_buckets(df_buckets)
    df_buckets_deduped = df_buckets_deduped.withColumn("Items", concat_ws(",", "Items"))
    save_dataframe_as_single_csv(df_buckets_deduped, "unique_bucket_itemsets.csv", "unique_buckets_temp")

    # Step 7: Stop Spark session
    spark.stop()
