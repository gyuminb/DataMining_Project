import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

cosine_sim = cosine_similarity(pivot_data)

# 유사도 매트릭스를 DataFrame으로 변환
similarity_df = pd.DataFrame(cosine_sim, index=pivot_data.index, columns=pivot_data.index)

# 특정 지갑 주소에 기반한 추천 함수 정의
def recommend_wallet(address, data, similarity_df, top_n=5):
    if address not in similarity_df.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # 지갑의 유사도 가져오기
    similar_wallets = similarity_df[address].sort_values(ascending=False).drop(address)  # 자기 자신 제외
    # 유사한 지갑들의 데이터 가져오기
    similar_wallets_data = data.loc[similar_wallets.index]

    # 추천 코인 계산 (유사 지갑의 보유량 평균)
    weighted_average = similar_wallets_data.T.dot(similar_wallets.values) / similar_wallets.sum()

    # 해당 지갑이 보유하지 않은 코인만 필터링
    existing_tokens = data.loc[address][data.loc[address] > 0].index
    recommendations = weighted_average.drop(existing_tokens).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0xd7f9f54194c633f36ccd5f3da84ad4a1c38cb2cb"  # 테스트용 지갑 주소
recommendations = recommend_wallet(address_to_recommend, pivot_data, similarity_df)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
