from SON_Apriori_FrequentItemsets_AssociationRule import SON_Apriori_FrequentItemsets_AssociationRule_analyze
from AssociationRule_Recommendation import get_wallet_portfolio, preprocess_user_portfolio_data, recommend_items_based_on_association_rule
from Collaborative_Filtering_Baseline_Predictor import preprocess_data, calculate_row_similarity, calculate_baseline_predictor, CF_baseline_predictor_userbased, CF_baseline_predictor_itembased
import os



if __name__ == "__main__":
    # 정제된 데이터를 대상으로 Apriori 기반의 SON 알고리즘(Spark 기반 Map Reduce 적용)으로 Frequent Itemsets & Association Rule 추출
    # SON_Apriori_FrequentItemsets_AssociationRule_analyze()
    
    # 사용자로부터 지갑 주소 입력받기
    user_address = input("Enter the user's wallet address: ").strip()
    association_rules_csv = input("Enter the path to the association rules CSV file: ").strip()

    # 스크래핑으로 포트폴리오 데이터 가져오기
    print(f"\nFetching portfolio for address: {user_address}")
    user_portfolio_data = get_wallet_portfolio(user_address)
    if not user_portfolio_data:
        print(f"Failed to retrieve portfolio data for address {user_address}. Exiting.")
        exit()
    
    # 포트폴리오 데이터 전처리
    portfolio_df = preprocess_user_portfolio_data(user_portfolio_data)

    # 사용자 포트폴리오 출력
    print("\nUser Portfolio Information:")
    print(portfolio_df[["Item", "Portfolio %", "Amount", "Value"]])
    
    # 사용자 포트폴리오 아이템 리스트 생성
    user_portfolio = portfolio_df['Item'].tolist()
    
    # Association Rule 기반 추천
    print("\nChecking for recommendations...")
    if not os.path.exists(association_rules_csv):
        print(f"Association Rules file not found: {association_rules_csv}")
        exit()

    recommended_items, applicable_rules = recommend_items_based_on_association_rule(user_portfolio, association_rules_csv)

    # 추천 결과 출력
    if recommended_items:
        print("\nRecommended Items:")
        for item in recommended_items:
            print(f" - {item}")
        
        # print("\nApplicable Association Rules:")
        # print(applicable_rules[["antecedent", "consequent", "confidence", "support"]])
        
    else:
        print("No recommendations available based on the current Association Rules.")

    file_path = './data/preprocessed_data.csv'
    pivot_data = preprocess_data(file_path, portfolio_df)
    
    row_similarity_df = calculate_row_similarity(pivot_data)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data)
    
    predicted_values = CF_baseline_predictor_userbased(user_address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5)
    sorted_predicted_values = dict(sorted(predicted_values.items(), key=lambda x: x[1], reverse=True))
    top_5_predictions = dict(list(sorted_predicted_values.items())[:5])
    
    # 출력
    print("\n\nTop 5 Predictions (user-based):")
    for item, value in top_5_predictions.items():
        print(f"{item}: {value}")

    row_similarity_df = calculate_row_similarity(pivot_data.T)
    global_mean, row_bias, col_bias = calculate_baseline_predictor(pivot_data.T)
    
    predicted_values = CF_baseline_predictor_itembased(user_address, pivot_data, row_similarity_df, global_mean, row_bias, col_bias, top_k=5)
    sorted_predicted_values = dict(sorted(predicted_values.items(), key=lambda x: x[1], reverse=True))
    top_5_predictions = dict(list(sorted_predicted_values.items())[:5])
    
    # 출력
    print("\n\nTop 5 Predictions (item-based):")
    for item, value in top_5_predictions.items():
        print(f"{item}: {value}")