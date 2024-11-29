from SON_Apriori_FrequentItemsets_AssociationRule import SON_Apriori_FrequentItemsets_AssociationRule_analyze
from AssociationRule_Recommendation import get_wallet_portfolio, preprocess_user_portfolio_data, recommend_items_based_on_association_rule
from Collaborative_Filtering_Baseline_Predictor import preprocess_data, calculate_row_similarity, calculate_baseline_predictor, CF_baseline_predictor_userbased, CF_baseline_predictor_itembased
import os, random, ast

def combine_recommendations_with_deduplication(association_rules, user_based_cf, item_based_cf, top_n = 10):
    final_scores = {}

    # Association Rule 기반 스코어 계산
    association_item_scores = {}
    for _, row in association_rules.iterrows():
        consequents = ast.literal_eval(row['consequent'])  # 추천 아이템 리스트
        if isinstance(consequents, str):  # 문자열인 경우
            consequents = [consequents]  # 리스트로 변환
        
        score = row['lift'] * row['confidence']
        for item in consequents:
            if item not in association_item_scores:
                association_item_scores[item] = score
            else:
                # 기존 값과 비교하여 최대값 유지
                association_item_scores[item] = max(association_item_scores[item], score)
    
    # Association Rule 결과를 final_scores에 반영
    for item, score in association_item_scores.items():
        final_scores[item] = final_scores.get(item, 0) + score * 2  # Association Rule 가중치 w1 = 1.5

    # User-Based CF 스코어 추가
    for item, score in user_based_cf.items():
        final_scores[item] = final_scores.get(item, 0) + score * 1.5  # User-Based CF 가중치 w2 = 1.5

    # Item-Based CF 스코어 추가
    for item, score in item_based_cf.items():
        final_scores[item] = final_scores.get(item, 0) + score * 1.75  # Item-Based CF 가중치 w3 = 1.75

    # 겹치는 아이템 보너스 점수 추가
    overlap_bonus = {}
    for item in set(final_scores.keys()):
        count = (item in association_item_scores) + \
                (item in user_based_cf) + \
                (item in item_based_cf)
        if count > 1:
            final_scores[item] += 10 * (count - 1)  # Overlap 보너스

    # 스코어 기반 정렬
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # 상위 N개 추천 (다양성 포함)
    diverse_recommendations = sorted_recommendations[:int(top_n * 0.8)] + \
                              random.sample(sorted_recommendations[int(top_n * 0.8):], k=int(top_n * 0.2))
    
    return diverse_recommendations


if __name__ == "__main__":
    # 정제된 데이터를 대상으로 Apriori 기반의 SON 알고리즘(Spark 기반 Map Reduce 적용)으로 Frequent Itemsets & Association Rule 추출
    # SON_Apriori_FrequentItemsets_AssociationRule_analyze()
    
    # 사용자로부터 지갑 주소 입력받기
    user_address = input("Enter the user's wallet address: ").strip()
    association_rules_csv = 'data/association_rules_full_base_0.05_0.5.csv'

    # 스크래핑으로 포트폴리오 데이터 가져오기
    print(f"\n[INFO] Fetching portfolio for address: {user_address}")
    user_portfolio_data = get_wallet_portfolio(user_address)
    if not user_portfolio_data:
        print(f"[ERROR] Failed to retrieve portfolio data for address {user_address}. Exiting.")
        exit()
    
    # 포트폴리오 데이터 전처리
    portfolio_df = preprocess_user_portfolio_data(user_portfolio_data)

    # 사용자 포트폴리오 출력
    print("\n[USER PORTFOLIO]")
    print(portfolio_df[["Item", "Portfolio %", "Amount", "Value"]].to_string(index=False))
    
    # 사용자 포트폴리오 아이템 리스트 생성
    user_portfolio = portfolio_df['Item'].tolist()
    
    # Association Rule 기반 추천
    print("\n[INFO] Checking for recommendations...")
    if not os.path.exists(association_rules_csv):
        print(f"[ERROR] Association Rules file not found: {association_rules_csv}")
        exit()

    recommended_items, applicable_rules = recommend_items_based_on_association_rule(user_portfolio, association_rules_csv)

    # 추천 결과 출력
    print("\n[ASSOCIATION RULE BASED RECOMMENDATIONS]")
    if recommended_items:
        for item in recommended_items:
            print(f" - {item}")
    else:
        print("No recommendations available based on the current Association Rules.")

    # Collaborative Filtering 기반 추천 (User-Based)
    print("\n[INFO] Calculating User-Based Collaborative Filtering Recommendations...")
    file_path = './data/preprocessed_data.csv'
    pivot_data = preprocess_data(file_path, portfolio_df)
    
    row_similarity_df = calculate_row_similarity(pivot_data)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data)
    
    user_predicted_values = CF_baseline_predictor_userbased(user_address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5)
    user_sorted_predicted_values = dict(sorted(user_predicted_values.items(), key=lambda x: x[1], reverse=True))
    user_top_5_predictions = dict(list(user_sorted_predicted_values.items())[:5])
    
    print("\n[USER-BASED TOP 5 PREDICTIONS]")
    for item, value in user_top_5_predictions.items():
        print(f" - {item}: {value:.4f}")

    # Collaborative Filtering 기반 추천 (Item-Based)
    print("\n[INFO] Calculating Item-Based Collaborative Filtering Recommendations...")
    row_similarity_df = calculate_row_similarity(pivot_data.T)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data.T)
    
    item_predicted_values = CF_baseline_predictor_itembased(user_address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5)
    item_sorted_predicted_values = dict(sorted(item_predicted_values.items(), key=lambda x: x[1], reverse=True))
    item_top_5_predictions = dict(list(item_sorted_predicted_values.items())[:5])
    
    print("\n[ITEM-BASED TOP 5 PREDICTIONS]")
    for item, value in item_top_5_predictions.items():
        print(f" - {item}: {value:.4f}")

    # 최종 추천 리스트
    diverse_recommendations = combine_recommendations_with_deduplication(applicable_rules, user_top_5_predictions, item_top_5_predictions, top_n = 10)
    
    print("\n[FINAL RECOMMENDATIONS]")
    print("Rank | Item                                         | Score")
    print("-----|---------------------------------------------|-------")
    for rank, (item, score) in enumerate(diverse_recommendations, start=1):
        print(f"{rank:<4} | {item:<45} | {score:.4f}")