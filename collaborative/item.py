import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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
    columns="Token",
    values="AmountRatio",
    aggfunc="sum",
    fill_value=0  # 비율이 없는 경우 0으로 채움
)

# 정규화 수행 (Min-Max Scaling)
scaler = MinMaxScaler()
normalized_pivot_data = pd.DataFrame(
    scaler.fit_transform(pivot_data),
    index=pivot_data.index,
    columns=pivot_data.columns
)

# 아이템-아이템 코사인 유사도 계산
item_cosine_sim = cosine_similarity(normalized_pivot_data.T)  # Transpose for item-based similarity
item_similarity_df = pd.DataFrame(item_cosine_sim, index=pivot_data.columns, columns=pivot_data.columns)

# 아이템-아이템 기반 추천 함수 정의
def recommend_items_item_based(address, data, item_similarity_df, top_n=5):
    """
    Recommend tokens for a wallet based on Item-Item Collaborative Filtering.
    """
    if address not in data.index:
        raise ValueError(f"지갑 주소 '{address}'가 데이터에 존재하지 않습니다.")

    # 사용자가 보유한 코인 및 보유 비율
    user_items = data.loc[address]

    # Step 1: 사용자가 보유한 코인들과 유사한 코인을 찾음
    item_scores = pd.Series(dtype=float)
    for item, ratio in user_items[user_items > 0].items():  # 보유 비율이 0보다 큰 코인만 처리
        similar_items = item_similarity_df[item]
        item_scores = item_scores.add(similar_items * ratio, fill_value=0)  # 유사도 * 보유 비율

    # Step 2: 이미 보유한 코인은 추천에서 제외
    existing_items = user_items[user_items > 0].index
    recommendations = item_scores.drop(existing_items).sort_values(ascending=False).head(top_n)

    return recommendations

# 테스트: 특정 지갑 주소에 대해 추천 코인 실행
address_to_recommend = "0x745869e92b46c5a4b959d5432ecc05a0b87d911a"  # 테스트용 지갑 주소
recommendations = recommend_items_item_based(
    address_to_recommend,
    pivot_data,
    item_similarity_df,
    top_n=15  # 추천할 코인의 수
)

# 결과 출력
print(f"추천 코인 리스트 (지갑 주소: {address_to_recommend}):")
print(recommendations)
