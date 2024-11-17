import pandas as pd
import itertools
from collections import defaultdict


def apriori(baskets, min_support):
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

# apriori 테스트
# CSV 데이터 로드 및 Apriori 실행
def run_apriori(input_csv, min_support):
    df = pd.read_csv(input_csv)
    baskets = df['Items'].apply(lambda x: x.split(",")).tolist()

    frequent_itemsets_with_counts = apriori(baskets, min_support)

    results_df = pd.DataFrame(frequent_itemsets_with_counts, columns=["items", "freq"])
    return results_df

# Apriori 실행
input_csv = "unique_bucket_itemsets.csv"
min_support = 0.05

frequent_itemsets_df = run_apriori(input_csv, min_support)
# freq 높은 순서로 정렬
frequent_itemsets_df = frequent_itemsets_df.sort_values(by="freq", ascending=False)
print(frequent_itemsets_df)

# 결과를 CSV로 저장
frequent_itemsets_df.to_csv("apriori_frequent_itemsets.csv", index=False)