from pyspark.sql import SparkSession
import pandas as pd
import itertools
from collections import defaultdict
import os, random, time

class SONAlgorithm:
    def __init__(self, itemsCol, minSupport, minConfidence):
        self.itemsCol = itemsCol
        self.minSupport = minSupport
        self.minConfidence = minConfidence
        self.totalbaskets = None
        self.frequent_itemsets = None
        self.association_rules = None
        
    def apriori(self, baskets, min_support):
        num_baskets = len(baskets)
        local_support = min_support * num_baskets

        # 1-Itemsets 찾기
        item_counts = defaultdict(int)
        for basket in baskets:
            for item in basket:
                item_counts[item] += 1
        L1 = {}
        for item, count in item_counts.items():
            if count >= local_support:
                L1[item] = count
        frequent_itemsets_with_counts = [(item, count) for item, count in L1.items()]
        
        # k-Itemsets 찾기
        k = 2
        current_Lk = set(L1.keys())  # L1으로 시작
        while current_Lk:
            #Ck 찾기
            if k == 2:
                pruned_candidates = set()
                for itemset in itertools.combinations(current_Lk, 2):
                    pruned_candidates.add(tuple(sorted(itemset)))
            else:
                candidates = set()
                for itemset in current_Lk:
                    for item in set(L1.keys()):
                        if item not in set(itemset):
                            union_set = tuple(sorted(set(itemset) | {item}))  # L1 아이템 하나 추가
                            if len(union_set) == k:
                                candidates.add(union_set)
                
                pruned_candidates = set()
                for candidate in candidates: # 모든 (k-1) 서브셋이 current_Lk에 있는지 확인
                    valid = True
                    for subset in itertools.combinations(candidate, k - 1):
                        if tuple(subset) not in current_Lk:             # current_Lk 집합에 있는지 확인
                            valid = False                       # 하나라도 없으면 False
                            break 

                    if valid:
                        pruned_candidates.add(candidate)  # 모든 조건을 만족하면 추가
            
            # Lk 찾기 
            itemset_counts = defaultdict(int)
            for basket in baskets:
                basket_set = set(basket)
                for candidate in pruned_candidates:
                    if set(candidate).issubset(basket_set):
                        itemset_counts[candidate] += 1
            
            current_Lk = set()
            for itemset, count in itemset_counts.items():
                if count >= local_support:
                    current_Lk.add(itemset)
                    frequent_itemsets_with_counts.append((list(itemset), count))
            # print(f"k-{k} current_Lk: {current_Lk}\n")
            k += 1

        return frequent_itemsets_with_counts  # itemset, freq 형태로 반환
    
    def generate_association_rules(self, rdd):
        if self.frequent_itemsets is None:
            print("No Valid frequent itemsets")
            return None
        
        rules = []
        rule_set = set()  # 중복 확인용
        support_cache = {}
        global_support = self.totalbaskets * self.minSupport
        
        frequent_itemsets = sorted(self.frequent_itemsets.to_dict("records"), key=lambda x: len(x['items']), reverse=True)
        
        for itemset_row in frequent_itemsets:
            itemset = itemset_row['items']
            itemset_support = itemset_row['freq']
            
            if len(itemset) == 1:
                continue
                
            # 모든 부분집합 생성
            subsets = list(itertools.chain(*[itertools.combinations(itemset, r) for r in range(1, len(itemset))]))
            
            for subset in subsets:
                # X -> Y, X ∪ Y는 frequent itemset  
                X = set(subset)
                Y = set(itemset) - X
                
                if not Y:
                    continue
                
                X_key = tuple(sorted(X))
                Y_key = tuple(sorted(Y))
                
                rule_str = f"X={X_key}, Y={Y_key}"    
                if rule_str not in rule_set:
                    # print(f"rule: {rule_str}")
                    
                    # X_support 캐시 확인 및 계산
                    if X_key in support_cache:
                        X_support = support_cache[X_key]
                    else:
                        X_support = rdd.filter(lambda basket: X.issubset(set(basket))).count()
                        support_cache[X_key] = X_support
                    
                    if X_support < global_support:
                        continue
                    
                    # Y_support 캐시 확인 및 계산
                    if Y_key in support_cache:
                        Y_support = support_cache[Y_key]
                    else:
                        Y_support = rdd.filter(lambda basket: Y.issubset(set(basket))).count()
                        support_cache[Y_key] = Y_support
                    
                    confidence = itemset_support / X_support if X_support > 0 else 0
                    lift = confidence / (Y_support / self.totalbaskets) if Y_support > 0 else 0
                    
                    if confidence > self.minConfidence:
                        rule_set.add(rule_str)
                        rules.append({
                            "antecedent": sorted(list(X)),
                            "consequent": sorted(list(Y)),
                            "confidence": confidence,
                            "lift": lift,
                            "support": itemset_support / self.totalbaskets
                        })
        self.association_rules = pd.DataFrame(rules)
    
    def fit(self, data):
        # SparkSession 생성
        spark = SparkSession.builder.appName("SON Algorithm").getOrCreate()
        # Spark RDD로 변환
        spark_data = spark.createDataFrame(data)
        # print("Spark DataFrame:")
        # spark_data.show(5)
        
        rdd = spark_data.select(self.itemsCol).rdd.map(lambda row: row[self.itemsCol])
        
        # num_partitions = rdd.getNumPartitions()
        # print(f"num_partitions: {num_partitions}")
        
        self.totalbaskets = rdd.count()
        global_support = self.totalbaskets * self.minSupport
        # print(f"totalbaskets: {self.totalbaskets}")
        # print(f"global_support: {global_support}")
        
        # 1-pass 각 파티션에서 Apriori로 candidate 찾기
        local_frequent_itemsets = rdd.mapPartitions(
            lambda partition: [
                (tuple(itemset) if isinstance(itemset, (list, tuple)) else (itemset,), count)
                for itemset, count in self.apriori(list(partition), self.minSupport)
            ]
        ).distinct()
        candidates = local_frequent_itemsets.map(lambda x: x[0]).distinct().collect()  # itemset만 가져오기
        # print("1-pass finished:")
        # print(candidates) 

        
        # 2-pass candidate count해서 frequent itemsets 찾기
        candidate_local_counts = rdd.flatMap(lambda basket : [(candidate, 1) for candidate in candidates if set(candidate).issubset(set(basket))])
        candidate_global_counts = candidate_local_counts.reduceByKey(lambda x,y : x+y)
        frequent_itemsets = candidate_global_counts.filter(lambda x: x[1] >= global_support)
        items = []
        freq = []
        
        for itemset, count in frequent_itemsets.collect():
            items.append(list(sorted(itemset)))
            freq.append(count)
        
        # Pandas DataFrame 생성
        self.frequent_itemsets = pd.DataFrame({"items": items, "freq": freq})
        print("starting generate association rules\n")
        self.generate_association_rules(rdd)
        
        spark.stop()

# Address 단위로 데이터를 Train/Test로 나누고 저장
def split_train_test_by_address(df, test_ratio=0.2, output_folder="data"):
    # Address 단위로 무작위 샘플링
    addresses = df['Address'].unique()
    test_size = int(len(addresses) * test_ratio)

    # Test Set에 사용할 Address 무작위 선택
    test_addresses = random.sample(list(addresses), test_size)
    train_addresses = list(set(addresses) - set(test_addresses))
    
    # Training/Test 데이터 분리
    train_df = df[df['Address'].isin(train_addresses)]
    test_df = df[df['Address'].isin(test_addresses)]
    
    # 데이터 저장
    train_path = os.path.join(output_folder, "training_set.csv")
    test_path = os.path.join(output_folder, "test_set.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training set saved to {train_path}, Test set saved to {test_path}")
    print(f"Training addresses: {len(train_addresses)}, Test addresses: {len(test_addresses)}")
    return train_df, test_df

def SON_Apriori_FrequentItemsets_AssociationRule_analyze():
    # min_support와 min_confidence 범위 설정
    min_support_values = [y / 100 for y in range(5, 21)]  # 0.05 ~ 0.2 (0.01 간격)
    min_confidence_values = [y / 100 for y in range(50, 81, 10)]  # 0.5 ~ 0.8 (0.1 간격)
    
    # 파일 경로 설정
    data_folder = "data"
    input_csv = os.path.join(data_folder, "unique_bucket_itemsets.csv")

    # 데이터 로드
    df = pd.read_csv(input_csv)
    df['Items'] = df['Items'].apply(lambda x: x.split(","))

    # 고유 식별자로 타임스탬프 추가
    timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    # Train/Test 분리
    perform_split = False  # True면 Train/Test 분리 실행, False면 전체 데이터에서 룰 생성
    if perform_split:
        
        # Training/Test 결과 저장 폴더 생성
        train_test_folder = os.path.join(data_folder, f"association_rules_train_test_{timestamp}")
        os.makedirs(train_test_folder, exist_ok=True)
        
        train_df, test_df = split_train_test_by_address(df, test_ratio=0.2, output_folder=train_test_folder)

        # SON Algorithm 실행 (최소 support와 confidence 값으로 한 번만 실행)
        base_support = min(min_support_values)
        base_confidence = min(min_confidence_values)
        print(f"\nRunning SON Algorithm on Training set with base support={base_support}, base confidence={base_confidence}")
        
        son_algo = SONAlgorithm(itemsCol="Items", minSupport=base_support, minConfidence=base_confidence)
        son_algo.fit(train_df)
        
        # 결과 저장
        sorted_frequent_itemsets = son_algo.frequent_itemsets.sort_values(by="freq", ascending=False)
        sorted_association_rules = son_algo.association_rules.sort_values(by="confidence", ascending=False)
        
        base_frequent_itemsets_csv = os.path.join(train_test_folder, f"frequent_itemsets_train_base_{base_support}.csv")
        base_association_rules_csv = os.path.join(train_test_folder, f"association_rules_train_base_{base_support}_{base_confidence}.csv")
        
        sorted_frequent_itemsets.to_csv(base_frequent_itemsets_csv, index=False)
        sorted_association_rules.to_csv(base_association_rules_csv, index=False)
        print(f"Base Frequent Itemsets saved to {base_frequent_itemsets_csv}")
        print(f"Base Association Rules saved to {base_association_rules_csv}")
        
        # 다양한 min_support와 min_confidence에 따라 필터링
        for min_support in min_support_values:
            # Support 기반 Frequent Itemsets 필터링 (freq 값 기준)
            filtered_itemsets = sorted_frequent_itemsets[
                sorted_frequent_itemsets["freq"] >= (min_support * len(train_df))  # freq >= support * total_baskets
            ]
            filtered_itemsets_csv = os.path.join(train_test_folder, f"frequent_itemsets_train_{min_support}.csv")
            filtered_itemsets.to_csv(filtered_itemsets_csv, index=False)
            print(f"Frequent Itemsets with support >= {min_support} saved to {filtered_itemsets_csv}")
        
        for min_support in min_support_values:
            for min_confidence in min_confidence_values:
                # Support와 Confidence 기반 Association Rules 필터링
                filtered_rules = sorted_association_rules[
                    (sorted_association_rules["confidence"] >= min_confidence) &
                    (sorted_association_rules["support"] >= min_support)
                ]
                filtered_rules_csv = os.path.join(train_test_folder, f"association_rules_train_{min_support}_{min_confidence}.csv")
                filtered_rules.to_csv(filtered_rules_csv, index=False)
                print(f"Association Rules with support >= {min_support} & confidence >= {min_confidence} saved to {filtered_rules_csv}")
    else:
        # 전체 데이터 기반으로 Association Rule 생성
        full_data_folder = os.path.join(data_folder, "association_rules_full")
        os.makedirs(full_data_folder, exist_ok=True)

        # SON Algorithm 실행 (최소 support와 confidence 값으로 한 번만 실행)
        base_support = min(min_support_values)
        base_confidence = min(min_confidence_values)
        print(f"\nRunning SON Algorithm on Full Dataset with base support={base_support}, base confidence={base_confidence}")

        son_algo = SONAlgorithm(itemsCol="Items", minSupport=base_support, minConfidence=base_confidence)
        son_algo.fit(df)
        
        # 결과 저장
        sorted_frequent_itemsets = son_algo.frequent_itemsets.sort_values(by="freq", ascending=False)
        sorted_association_rules = son_algo.association_rules.sort_values(by="confidence", ascending=False)
        
        base_frequent_itemsets_csv = os.path.join(full_data_folder, f"frequent_itemsets_full_base_{base_support}.csv")
        base_association_rules_csv = os.path.join(full_data_folder, f"association_rules_full_base_{base_support}_{base_confidence}.csv")
        
        sorted_frequent_itemsets.to_csv(base_frequent_itemsets_csv, index=False)
        sorted_association_rules.to_csv(base_association_rules_csv, index=False)
        print(f"Base Frequent Itemsets saved to {base_frequent_itemsets_csv}")
        print(f"Base Association Rules saved to {base_association_rules_csv}")
        
        # 다양한 min_support와 min_confidence에 따라 필터링
        for min_support in min_support_values:
            # Support 기반 Frequent Itemsets 필터링 (freq 값 기준)
            filtered_itemsets = sorted_frequent_itemsets[
                sorted_frequent_itemsets["freq"] >= (min_support * len(df))  # freq >= support * total_baskets
            ]
            filtered_itemsets_csv = os.path.join(full_data_folder, f"frequent_itemsets_full_{min_support}.csv")
            filtered_itemsets.to_csv(filtered_itemsets_csv, index=False)
            print(f"Frequent Itemsets with support >= {min_support} saved to {filtered_itemsets_csv}")
        
        for min_support in min_support_values:
            for min_confidence in min_confidence_values:
                # Support와 Confidence 기반 Association Rules 필터링
                filtered_rules = sorted_association_rules[
                    (sorted_association_rules["confidence"] >= min_confidence) &
                    (sorted_association_rules["support"] >= min_support)
                ]
                filtered_rules_csv = os.path.join(full_data_folder, f"association_rules_full_{min_support}_{min_confidence}.csv")
                filtered_rules.to_csv(filtered_rules_csv, index=False)
                print(f"Association Rules with support >= {min_support} & confidence >= {min_confidence} saved to {filtered_rules_csv}")