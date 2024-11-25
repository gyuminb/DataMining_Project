import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, collect_list, concat_ws, posexplode, count, array_distinct

def ensure_data_folder(folder="data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

def load_data(spark, input_file):
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    return df

def save_dataframe_as_single_csv(df, output_file, temp_dir):
    df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")
    temp_file = [f for f in os.listdir(temp_dir) if f.startswith("part-")][0]
    shutil.move(os.path.join(temp_dir, temp_file), output_file)
    shutil.rmtree(temp_dir)

def preprocess_initial_data(spark, input_file, output_file, temp_dir="preprocessed_temp"):
    # 데이터 로드
    ensure_data_folder()
    df = load_data(spark, input_file)

    # Name Tag가 없는 지갑 필터링
    df_filtered = df.filter(col("Name Tag").isNull())

    # Contract Address와 Address가 동일한 경우 Contract Address를 null로 변경
    df_filtered = df_filtered.withColumn(
        "Contract Address",
        when(col("Address") == col("Contract Address"), None).otherwise(col("Contract Address"))
    )

    # (Chain, Token, Contract Address) 아이템 생성
    df_with_items = df_filtered.withColumn("Item", concat_ws("_", col("Chain"), col("Token"), col("Contract Address")))

    # 'ETH_Ether (ETH)' 제거 (Chain이 ETH, Token이 Ether (ETH), Contract Address가 NULL인 경우)
    df_preprocessed = df_with_items.filter(~(
            (col("Chain") == "ETH") &
            (col("Token") == "Ether (ETH)") &
            (col("Contract Address").isNull())
    ))

    # Preprocessed 데이터셋 저장
    save_dataframe_as_single_csv(df_preprocessed, output_file, temp_dir)

    # 최종 전처리된 지갑 주소 개수 출력
    num_wallets = df_preprocessed.select("Address").distinct().count()
    print(f"Final preprocessed data's Number of wallets (without Name Tag and Ether): {num_wallets}")
    print(f"Preprocessed data saved to {output_file}.\n")


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


def process_bucket_itemsets(spark, input_file):
    # 데이터 로드
    ensure_data_folder()
    df = load_data(spark, input_file)

    # 지갑 주소별 아이템 그룹화
    df_buckets = group_items_by_address(df)

    # 중복된 아이템 찾기
    df_duplicates = find_duplicate_items(df_buckets)
    duplicates_output_path = os.path.join("data", "duplicate_items.csv")
    save_dataframe_as_single_csv(df_duplicates, duplicates_output_path, "duplicates_temp")
    print(f"Duplicate items saved to '{duplicates_output_path}'.")

    # 중복을 제거한 Bucket-Itemset 생성 및 저장
    df_buckets_deduped = remove_duplicates_in_buckets(df_buckets)
    df_buckets_deduped = df_buckets_deduped.withColumn("Items", concat_ws(",", "Items"))
    deduped_output_path = os.path.join("data", "unique_bucket_itemsets.csv")
    save_dataframe_as_single_csv(df_buckets_deduped, deduped_output_path, "unique_buckets_temp")

    # 중복 제거된 지갑 주소 수 출력
    unique_wallet_count = df_buckets_deduped.select("Address").distinct().count()
    print(f"Unique bucket itemsets saved to '{deduped_output_path}'.")
    print(f"Total number of unique wallets: {unique_wallet_count}")

if __name__ == "__main__":
    # Step 1: SparkSession 생성
    spark = SparkSession.builder.appName("Etherscan Data Processing").getOrCreate()

    # Step 2: 초기 공통 전처리 수행
    raw_input_file = os.path.join("data", "etherscan_merged_data.csv")
    preprocessed_file = os.path.join("data", "preprocessed_data.csv")
    preprocess_initial_data(spark, raw_input_file, preprocessed_file)

    # Step 3: 전처리된 데이터 기반으로 Bucket-Itemset 처리
    process_bucket_itemsets(spark, preprocessed_file)

    # Step 4: SparkSession 종료
    spark.stop()