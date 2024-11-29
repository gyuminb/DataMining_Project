import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# CSV 데이터 불러오기
file_path = './preprocessed_data.csv'
data = pd.read_csv(file_path)

# 'Amount' 열에서 쉼표 제거 및 숫자로 변환
data['Amount'] = data['Amount'].replace({',': ''}, regex=True).astype(float)

# 각 지갑 주소별 총 보유량 계산
data['TotalAmount'] = data.groupby('Address')['Amount'].transform('sum')

# 각 지갑 주소와 코인의 보유 비율 계산
data['AmountRatio'] = data['Amount'] / data['TotalAmount']

# 피벗 테이블 생성: 지갑 주소를 행(Index), 코인을 열(Columns), 보유 비율(AmountRatio)을 값으로
pivot_data = data.pivot_table(
    index="Address",
    columns="Item",
    values="AmountRatio",
    aggfunc="sum",
    fill_value=0  # 비율이 없는 경우 0으로 채움
)



# 아이템-아이템 코사인 유사도 계산
item_cosine_sim = cosine_similarity(pivot_data.T)  # Transpose for item-based similarity
item_similarity_df = pd.DataFrame(item_cosine_sim, index=pivot_data.columns, columns=pivot_data.columns)

# Baseline Predictor 계산 함수
def calculate_baseline_predictor(pivot_data):
    """
    Calculate global average, user bias, and item bias for Baseline Predictor.
    """
    # Global average (μ)
    global_mean = pivot_data.values.mean()

    # User bias (b_u)
    user_bias = pivot_data.mean(axis=1) - global_mean

    # Item bias (b_i)
    item_bias = pivot_data.mean(axis=0) - global_mean

    return global_mean, user_bias, item_bias

# Baseline Predictor 계산
global_mean, user_bias, item_bias = calculate_baseline_predictor(pivot_data)

# 추천 함수: 변화량(delta)을 반영한 Item-Item 협업 필터링
def recommend_items_item_based(address, data, item_similarity_df, global_mean, user_bias, item_bias, top_n=5, k=5):
    """
    Recommend tokens using Baseline Predictor with delta modeling for Item-Item Collaborative Filtering.
    """
    if address not in data.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # 사용자가 보유한 토큰 및 보유 비율 가져오기
    user_items = data.loc[address]

    # Step 1: 각 토큰에 대해 점수 예측
    weighted_scores = {}
    for target_token in data.columns:
        # Baseline 점수 계산: b_ui = μ + b_u + b_i
        baseline_score = global_mean + user_bias[address] + item_bias[target_token]

        # 유사한 아이템 선택
        similar_items = item_similarity_df[target_token].sort_values(ascending=False).drop(target_token).head(k)

        # 협업 필터링 점수 계산 (변화량 기반)
        delta_scores = [
            (user_items[item] - (global_mean + user_bias[address] + item_bias[item])) * similarity
            for item, similarity in similar_items.items()
        ]
        delta_average = sum(delta_scores) / similar_items.sum() if similar_items.sum() > 0 else 0

        # 최종 점수: Baseline + 변화량 기반 협업 필터링 점수
        weighted_scores[target_token] = baseline_score + delta_average

    # Step 2: 사용자가 이미 보유한 토큰 제외
    existing_items = user_items[user_items > 0].index
    recommendations = pd.Series(weighted_scores).drop(existing_items).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x745869e92b46c5a4b959d5432ecc05a0b87d911a"  # 테스트용 지갑 주소
recommendations = recommend_items_item_based(
    address_to_recommend,
    pivot_data,
    item_similarity_df,
    global_mean,
    user_bias,
    item_bias,
    top_n=5,
    k=5  # 유사 아이템 수
)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
