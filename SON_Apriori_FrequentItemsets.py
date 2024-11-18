from pyspark.sql import SparkSession
import pandas as pd
import itertools
from collections import defaultdict
import os

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
    
if __name__ == "__main__":
    # min_support와 min_confidence 범위 설정
    min_support_values = [y / 100 for y in range(5, 11)]  # 0.05 ~ 0.1 (0.01 간격)
    min_confidence_values = [y / 100 for y in range(50, 81, 10)]  # 0.5 ~ 0.8 (0.1 간격)
    
    # 파일 경로 설정
    data_folder = "data"
    input_csv = os.path.join(data_folder, "unique_bucket_itemsets.csv")

    # 데이터 로드
    df = pd.read_csv(input_csv)
    df['Items'] = df['Items'].apply(lambda x: x.split(","))

    # 반복 실행
    for min_support in min_support_values:
        for min_confidence in min_confidence_values:
            print(f"\nRunning SON Algorithm with min_support={min_support}, min_confidence={min_confidence}")
            
            # 결과 파일 경로 설정
            frequent_itemsets_output_csv = os.path.join(data_folder, f"frequent_itemsets_son_{min_support}.csv")
            association_rules_output_csv = os.path.join(data_folder, f"association_rules_son_{min_support}_{min_confidence}.csv")
            
            # apriori spark 기반 SON Algorithm 실행
            son_algo = SONAlgorithm(itemsCol="Items", minSupport=min_support, minConfidence=min_confidence)
            son_algo.fit(df)

            # 결과 출력 및 저장
            sorted_frequent_itemsets = son_algo.frequent_itemsets.sort_values(by="freq", ascending=False)
            sorted_association_rules = son_algo.association_rules.sort_values(by="confidence", ascending=False)
            
            print("Frequent Itemsets (sorted by freq):")
            print(sorted_frequent_itemsets)
            print("\nAssociation Rules (sorted by confidence):")
            print(sorted_association_rules)

            sorted_frequent_itemsets.to_csv(frequent_itemsets_output_csv, index=False)
            sorted_association_rules.to_csv(association_rules_output_csv, index=False)
            print(f"Frequent itemsets saved to {frequent_itemsets_output_csv}")
            print(f"Association rules saved to {association_rules_output_csv}")