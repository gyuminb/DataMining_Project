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



# 코사인 유사도 계산
cosine_sim = cosine_similarity(pivot_data)

# 유사도 매트릭스를 DataFrame으로 변환
similarity_df = pd.DataFrame(cosine_sim, index=pivot_data.index, columns=pivot_data.index)

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

# 추천 함수: 변화량(delta)을 반영한 협업 필터링
def recommend_wallet_with_delta(address, data, similarity_df, global_mean, user_bias, item_bias, top_n=5, k=5, min_similarity=0.5):
    """
    Recommend tokens using Baseline Predictor with delta modeling.
    """
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # Step 1: 유사도 임계값을 기반으로 k명의 가장 유사한 사용자 선택
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address)
    similar_wallets = similar_wallets[similar_wallets > min_similarity].head(k)

    # Step 2: 각 토큰의 점수 예측
    weighted_scores = {}
    for token in data.columns:
        # Baseline 점수 계산: b_ui = μ + b_u + b_i
        baseline_score = global_mean + user_bias[address] + item_bias[token]

        # 협업 필터링 점수 계산 (변화량 기반)
        delta_scores = [
            (data.loc[user, token] - (global_mean + user_bias[user] + item_bias[token])) * similarity
            for user, similarity in similar_wallets.items()
        ]
        delta_average = sum(delta_scores) / similar_wallets.sum() if similar_wallets.sum() > 0 else 0

        # 최종 점수: Baseline + 변화량 기반 협업 필터링 점수
        weighted_scores[token] = baseline_score + delta_average

    # Step 3: 해당 지갑이 이미 보유한 토큰 제외
    existing_tokens = data.loc[address][data.loc[address] > 0].index
    recommendations = pd.Series(weighted_scores).drop(existing_tokens).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x745869e92b46c5a4b959d5432ecc05a0b87d911a"  # 테스트용 지갑 주소
recommendations = recommend_wallet_with_delta(
    address_to_recommend,
    pivot_data,
    similarity_df,
    global_mean,
    user_bias,
    item_bias,
    top_n=5,
    k=5,  # 유사 사용자 수
    min_similarity=0.5  # 유사도 임계값
)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)


# 특정 Item의 점수와 순위를 반환하는 함수
def get_item_rank(address, item, data, similarity_df, global_mean, user_bias, item_bias, k=5, min_similarity=0.5):
    """
    Get the score and rank of a specific item for a given address.
    """
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")
    if item not in data.columns:
        raise ValueError(f"아이템 '{item}'이 데이터에 존재하지 않습니다.")

    # Step 1: 모든 아이템 점수 계산
    all_scores = {}
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address)
    similar_wallets = similar_wallets[similar_wallets > min_similarity].head(k)

    for current_item in data.columns:
        # Baseline 점수 계산
        baseline_score = global_mean + user_bias[address] + item_bias[current_item]

        # 협업 필터링 점수 계산 (변화량 기반)
        delta_scores = [
            (data.loc[user, current_item] - (global_mean + user_bias[user] + item_bias[current_item])) * similarity
            for user, similarity in similar_wallets.items()
        ]
        delta_average = sum(delta_scores) / similar_wallets.sum() if similar_wallets.sum() > 0 else 0

        # 최종 점수 계산
        final_score = baseline_score + delta_average
        if abs(final_score) < 1e-10:  # 작은 값은 0으로 처리
            final_score = 0
        all_scores[current_item] = final_score

    # Step 2: 점수를 기준으로 정렬하여 순위 계산
    sorted_scores = pd.Series(all_scores).sort_values(ascending=False)
    rank = sorted_scores.rank(ascending=False)

    # Step 3: 원하는 Item의 점수와 순위 반환
    if item not in sorted_scores.index:
        raise ValueError(f"아이템 '{item}'이 점수 계산 결과에 존재하지 않습니다.")
    item_score = sorted_scores[item]
    item_rank = rank[item]

    return item_score, item_rank

# 테스트: 특정 Item의 점수와 순위 계산
address_to_recommend = "0x745869e92b46c5a4b959d5432ecc05a0b87d911a"  # 테스트용 지갑 주소
item_to_check = "ETH_Ducky (DUCKY)_0x699ec925118567b6475fe495327ba0a778234aaa"  # 테스트용 아이템 이름

item_score, item_rank = get_item_rank(
    address_to_recommend,
    item_to_check,
    pivot_data,
    similarity_df,
    global_mean,
    user_bias,
    item_bias,
    k=5,  # 유사 사용자 수
    min_similarity=0.5  # 유사도 임계값
)

# 결과 출력
print(f"지갑 주소: {address_to_recommend}, 아이템: {item_to_check}")
print(f"점수: {item_score}, 순위: {int(item_rank)}")

