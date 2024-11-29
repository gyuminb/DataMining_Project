import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# CSV 데이터 불러오기
file_path = './preprocessed_data.csv'
data = pd.read_csv(file_path)

# 'Amount' 열에서 쉼표 제거 및 숫자로 변환
data['Amount'] = data['Amount'].replace({',': ''}, regex=True).astype(float)

# 피벗 테이블 생성: 지갑 주소를 행(Index), 코인을 열(Columns), 보유량(Amount)을 값으로
pivot_data = data.pivot_table(
    index="Address",
    columns="Token",
    values="Amount",
    aggfunc="sum",
    fill_value=0
)

# 보유 비율 계산: 각 지갑의 총 보유량으로 나누기
pivot_data_ratio = pivot_data.div(pivot_data.sum(axis=1), axis=0)

# 정규화 수행 (Min-Max Scaling)
scaler = MinMaxScaler()
normalized_pivot_data = pd.DataFrame(
    scaler.fit_transform(pivot_data_ratio),
    index=pivot_data.index,
    columns=pivot_data.columns
)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(normalized_pivot_data)

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
global_mean, user_bias, item_bias = calculate_baseline_predictor(pivot_data_ratio)

# 추천 함수: 최적 조건 반영
def recommend_wallet(address, data, similarity_df, global_mean, user_bias, item_bias, top_n=5, k=5, min_similarity=0.5, alpha=0.7):
    """
    Recommend tokens with optimal conditions using Baseline Predictor and cosine similarity.
    """
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # Step 1: 유사도 임계값을 기반으로 k명의 가장 유사한 사용자 선택
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address)
    similar_wallets = similar_wallets[similar_wallets > min_similarity].head(k)

    # Step 2: 각 토큰의 점수 예측
    weighted_scores = {}
    for token in data.columns:
        # Baseline Predictor: μ + b_u + b_i
        baseline_score = global_mean + user_bias[address] + item_bias[token]

        # 협업 필터링 기반 점수: 가중 평균
        scores = [
            (data.loc[user, token] * similarity)  # 보유 비율 * 유사도
            for user, similarity in similar_wallets.items()
        ]
        weighted_average = sum(scores) / similar_wallets.sum() if similar_wallets.sum() > 0 else 0

        # 최종 점수: α * Baseline + (1 - α) * 가중 평균
        weighted_scores[token] = alpha * baseline_score + (1 - alpha) * weighted_average

    # Step 3: 해당 지갑이 이미 보유한 토큰 제외
    existing_tokens = data.loc[address][data.loc[address] > 0].index
    recommendations = pd.Series(weighted_scores).drop(existing_tokens).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x062a31bd836cecb1b6bc82bb107c8940a0e6a01d"  # 테스트용 지갑 주소
recommendations = recommend_wallet(
    address_to_recommend,
    pivot_data_ratio,
    similarity_df,
    global_mean,
    user_bias,
    item_bias,
    top_n=5,
    k=5,  # 유사 사용자 수
    min_similarity=0.5,  # 유사도 임계값
    alpha=0.7  # Baseline vs. CF 점수 비율
)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
