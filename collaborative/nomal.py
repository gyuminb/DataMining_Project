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

# 정규화 수행 (Min-Max Scaling)
scaler = MinMaxScaler()
normalized_pivot_data = pd.DataFrame(
    scaler.fit_transform(pivot_data),
    index=pivot_data.index,
    columns=pivot_data.columns
)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(normalized_pivot_data)

# 유사도 매트릭스를 DataFrame으로 변환
similarity_df = pd.DataFrame(cosine_sim, index=pivot_data.index, columns=pivot_data.index)

# 특정 지갑 주소에 기반한 추천 함수 정의
def recommend_wallet(address, data, similarity_df, top_n=5, k=3):
    """
    Recommend tokens based on User-User CF using cosine similarity.
    """
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # Step 1: k명의 가장 유사한 사용자 선택
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address).head(k)

    # Step 2: 각 토큰의 점수 예측 (가중 평균)
    weighted_scores = {}
    for token in data.columns:
        # 유사한 사용자들의 보유량과 유사도를 사용하여 가중 평균 계산
        scores = [
            (data.loc[user, token] * similarity)  # 보유량 * 유사도
            for user, similarity in similar_wallets.items()
        ]
        # 점수 계산 및 저장
        weighted_scores[token] = sum(scores) / similar_wallets.sum()

    # Step 3: 해당 지갑이 이미 보유한 토큰 제외
    existing_tokens = data.loc[address][data.loc[address] > 0].index
    recommendations = pd.Series(weighted_scores).drop(existing_tokens).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x062a31bd836cecb1b6bc82bb107c8940a0e6a01d"  # 테스트용 지갑 주소
recommendations = recommend_wallet(address_to_recommend, normalized_pivot_data, similarity_df)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
